import os
import re
import struct

import numpy as np

from utils.data import read_binary
from utils.project import get_project_root


def add_weighted_avg(average, weight, new_sample, new_weight):
    new_average = (weight * average + new_weight * new_sample) / (weight + new_weight)
    return new_average


class Coordinate:
    def __init__(self, cx, cy):
        self.cx = float(cx)
        self.cy = float(cy)

    def __str__(self):
        return "X: {:10.6f}, Y: {:10.6f}".format(self.cx, self.cy)

    def __neg__(self):
        return Coordinate(-self.cx, -self.cy)

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Coordinate(self.cx + other, self.cy + other)
        elif isinstance(other, Coordinate):
            return Coordinate(self.cx + other.cx, self.cy + other.cy)
        else:
            return Coordinate(self.cx, self.cy)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Coordinate(self.cx * other, self.cy * other)
        elif isinstance(other, Coordinate):
            return Coordinate(self.cx * other.cx, self.cy * other.cy)
        else:
            return Coordinate(self.cx, self.cy)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Coordinate(self.cx / other, self.cy / other)
        elif isinstance(other, Coordinate):
            return Coordinate(self.cx / other.cx, self.cy / other.cy)
        else:
            return Coordinate(self.cx, self.cy)

    def __pow__(self, power, modulo=None):
        return Coordinate(self.cx**power, self.cy**power)

    def __eq__(self, other):
        return abs(self.cx - other.cx) < 1e-8 and abs(self.cy - other.cy) < 1e-8

    def __abs__(self):
        return (self.cx * self.cx + self.cy + self.cy)**0.5

    def __hash__(self):
        return hash((self.cx, self.cy))

    def transpose(self):
        return Coordinate(self.cy, self.cx)

    def conj(self):
        return Coordinate(self.cx, -self.cy)

    def hilbert(self):
        return Coordinate(self.cy, -self.cx)

    def sum(self):
        return self.cx + self.cy


def determinant(coordi1, coordi2):
    return coordi1.cx * coordi2.cy - coordi1.cy * coordi2.cx


def inner_product(coordi1, coordi2):
    return coordi1.cx * coordi2.cx + coordi1.cy * coordi2.cy


def distance(coordi1, coordi2):
    return (((coordi2 - coordi1)**2.0).sum())**0.5


class Trace:
    def __init__(self, data, coordi):
        self.data = data
        self.nz = self.data.shape[0]
        self.coordi = coordi

    def set_data(self, data):
        assert len(data.shape) == 1
        self.data = data
        self.nz = self.data.shape[0]

    def set_coordi(self, coordi):
        self.coordi = coordi


def detect_stationary_ends(coordi):
    nt = len(coordi)

    if nt == 1:
        return 0, nt

    coordi_diff = np.diff(coordi, axis=0)
    coordi_diff = (coordi_diff[:, 0]**2.0 + coordi_diff[:, 1]**2.0)**0.5
    change_indices = np.where(coordi_diff >= 0.5)[0]

    if len(change_indices) == 0:
        return nt // 2, nt // 2 + 1

    first_move_idx = change_indices[0]
    last_move_idx = change_indices[-1]

    start_slice = first_move_idx
    end_slice = last_move_idx + 2

    return start_slice, end_slice


# This subroutine must be modified according to the SEG-Y file format.
# Shell script in the data directory need to be executed before this subroutine to extract .head, .suhead, .su, and .bin files
def convert_line_segy_to_numpy(data_pool="southsea", tag=None, is_align_coordi=False, resample=1, sz=None, ez=None):
    data_root = os.path.join(get_project_root(), "data", data_pool)
    suhead_path = os.path.join(data_root, tag + ".suhead")
    header_path = os.path.join(data_root, tag + ".head")
    binary_path = os.path.join(data_root, tag + ".bin")

    # "amp" variable indicate the possibly largest amplitude value, need to be modified according to the SEG-Y file
    with open(header_path, encoding="latin-1") as f:
        for _ in range(12):
            f.readline()

        amp_line = f.readline()
        amp_min = float(re.findall(r"[-+]?(?:\d*\.*\d+)", amp_line.split()[4])[0])
        amp_max = float(re.findall(r"[-+]?(?:\d*\.*\d+)", amp_line.split()[6])[0])
        amp = max(-amp_min, amp_max)

        # Examples
        # amp = 1.0
        # amp = 32767

    with open(suhead_path, "rb") as suhead_file:
        suhead = suhead_file.read()

        file_size = len(suhead)
        nt = file_size // 240

        fth = suhead[:240]
        nz = struct.unpack("h", fth[114: 116])[0]
        dz = struct.unpack("h", fth[116: 118])[0] / 1000000

        trace_coordi = np.zeros((nt, 2), dtype=np.float64)
        if is_align_coordi:  # Find the closest line segment to the traces' coordinates
            b = np.zeros((2 * nt, 1), dtype=np.float64)
            A = np.zeros((2 * nt, 4), dtype=np.float64)
            for it in range(nt):
                th = suhead[240 * it: 240 * (it + 1)]
                cx = struct.unpack("i", th[72: 76])[0]
                cy = struct.unpack("i", th[76: 80])[0]

                b[2 * it + 0, 0] = cx
                A[2 * it + 0, 0] = (nt - 1 - it) / (nt - 1)
                A[2 * it + 0, 2] = it / (nt - 1)

                b[2 * it + 1, 0] = cy
                A[2 * it + 1, 1] = (nt - 1 - it) / (nt - 1)
                A[2 * it + 1, 3] = it / (nt - 1)
            At = np.transpose(A)
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(At, A)), At), b)
            scoordi = Coordinate(x[0], x[1])
            ecoordi = Coordinate(x[2], x[3])
            for it in range(nt):
                coordi = scoordi + (ecoordi - scoordi) * (it / (nt - 1))
                trace_coordi[it] = [coordi.cx, coordi.cy]
        else:
            for it in range(nt):
                th = suhead[240 * it: 240 * (it + 1)]
                cx = struct.unpack("i", th[72: 76])[0]
                cy = struct.unpack("i", th[76: 80])[0]
                trace_coordi[it] = [cx, cy]

    data = read_binary(binary_path, nz)
    sz = 0  if sz is None else sz
    ez = nz if ez is None else ez
    rsz = max(sz, 0)
    rez = min(ez, nz)
    data = data[rsz: rez]
    if resample != 1:
        resample = int(resample)
        data = data[::resample]
        dz = dz * resample
    nz = data.shape[0]

    # Remove stationary traces at the both ends of 2D seismic data
    start_it, end_it = detect_stationary_ends(trace_coordi)
    data = data[:, start_it: end_it]
    trace_coordi = trace_coordi[start_it: end_it]
    nt = end_it - start_it

    np.save(os.path.join(data_root, tag), data)
    with open(os.path.join(data_root, tag + ".nphead"), "w") as f:
        f.write("shape\t" + str(nz) + "\t" + str(nt) + "\n")
        f.write("amp\t" + str(amp) + "\n")
        f.write("dz\t" + str(dz) + "\n")
        f.write("\n" + "trace coordi" + "\n")
        for it in range(nt):
            f.write(str(it) + "\t" + str(trace_coordi[it, 0]) + "\t" + str(trace_coordi[it, 1]) + "\n")


# This class assumes a 2D seismic line is almost straight
# To conform with curved lines, this class need to be modified
class SeismicLine:
    def __init__(self, data_pool="southsea", tag=None):
        if tag is None:
            self.amp = 1.0
            self.dz = 1.0
            self.clean_data(shape=(1, 1))
            self.scoordi = Coordinate(0.0, 0.0)
            self.ecoordi = Coordinate(0.0, 0.0)
        else:
            data_root = os.path.join(get_project_root(), "data", data_pool)
            head_path = os.path.join(data_root, tag + ".nphead")
            data_path = os.path.join(data_root, tag + ".npy")

            self.read_nphead(head_path)
            self.data = np.load(data_path)
            self.data = self.data / self.amp
            if self.shape != self.data.shape:
                print(self.shape, self.data.shape)
                print("Data shape has error!")
                exit(-1)
            self.dp = np.zeros(self.shape[1:], dtype=np.float32)

    def read_nphead(self, header_path):
        with open(header_path, encoding="latin-1") as f:
            shape_line = f.readline()
            nz = int(shape_line.split()[1])
            nt = int(shape_line.split()[2])
            self.shape = (nz, nt)

            amp_line = f.readline()
            self.amp = float(amp_line.split()[1])

            dz_line = f.readline()
            self.dz = float(dz_line.split()[1])

            for _ in range(2):
                f.readline()

            b = np.zeros((2 * nt, 1), dtype=np.float64)
            A = np.zeros((2 * nt, 4), dtype=np.float64)
            for _ in range(nt):
                coordi_line = f.readline()
                it = int(coordi_line.split()[0])
                cx = float(coordi_line.split()[1])
                cy = float(coordi_line.split()[2])

                b[2 * it + 0, 0] = cx
                A[2 * it + 0, 0] = (nt - 1 - it) / (nt - 1)
                A[2 * it + 0, 2] = it / (nt - 1)

                b[2 * it + 1, 0] = cy
                A[2 * it + 1, 1] = (nt - 1 - it) / (nt - 1)
                A[2 * it + 1, 3] = it / (nt - 1)
            At = np.transpose(A)
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(At, A)), At), b)
            self.scoordi = Coordinate(x[0], x[1])
            self.ecoordi = Coordinate(x[2], x[3])

    def clean_data(self, shape=None):
        if shape is None:
            self.data[:, :] = 0.0
            self.dp = np.zeros(self.shape[1:], dtype=np.float32)
        else:
            self.data = np.zeros(shape, dtype=np.float32)
            self.shape = shape
            self.dp = np.zeros(self.shape[1:], dtype=np.float32)

    def set_amp(self, amp=None):
        self.amp = amp if amp is not None else np.max(np.abs(self.data))
        self.data = self.data / self.amp

    def set_coordi(self, scoordi=None, ecoordi=None):
        self.scoordi = scoordi if scoordi is not None else self.scoordi
        self.ecoordi = ecoordi if ecoordi is not None else self.ecoordi

    def cut_zaxis(self, sz, ez):
        rsz = max(sz, 0)
        rez = min(ez, self.shape[0])

        self.data = self.data[rsz: rez]
        self.shape = self.data.shape

    def idx_to_coordi(self, it):
        return self.scoordi + (self.ecoordi - self.scoordi) * (it / (self.shape[1] - 1))

    def coordi_to_idx(self, coordi):
        grad = self.ecoordi - self.scoordi
        a = -grad.cy
        b = grad.cx
        ab_sqsum = a * a + b * b
        c = -(a * self.scoordi.cx + b * self.scoordi.cy)
        distance = (a * coordi.cx + b * coordi.cy + c) / ab_sqsum**0.5
        closest = Coordinate(
            (b * (b * coordi.cx - a * coordi.cy) - a * c) / ab_sqsum,
            (a * (a * coordi.cy - b * coordi.cx) - b * c) / ab_sqsum
        )
        if grad.cx != 0:
            it = (self.shape[1] - 1) * ((closest.cx - self.scoordi.cx) / grad.cx)
        else:
            it = (self.shape[1] - 1) * ((closest.cy - self.scoordi.cy) / grad.cy)
        return it, closest, distance

    def get_trace_idx(self, it):
        if it < 0 or it > self.shape[1] - 1:
            trace = np.zeros(self.shape[0], dtype=np.float32)
        else:
            if abs(round(it) - it) < 0.01:
                trace = self.data[:, int(round(it))]
            else:
                portion = it - np.floor(it)
                pre_trace = self.data[:, int(np.floor(it))]
                pro_trace = self.data[:, int(np.ceil(it))]
                trace = portion * pro_trace + (1.0 - portion) * pre_trace
        return trace

    def get_trace_coordi(self, coordi):
        it, _, _ = self.coordi_to_idx(coordi)
        return self.get_trace_idx(it)

    def insert_trace_idx(self, new_trace, it):
        assert len(new_trace) == self.shape[0]
        if 0 <= it <= self.shape[1] - 1:
            if abs(round(it) - it) < 0.01:
                trace = self.data[:, int(round(it))]
                weight = self.dp[int(round(it))]
                self.data[:, int(round(it))] = add_weighted_avg(trace, weight, new_trace, 1.0)
                self.dp[int(round(it))] += 1.0
            else:
                portion = it - np.floor(it)

                pre_trace = self.data[:, int(np.floor(it))]
                pre_weight = self.dp[int(np.floor(it))]
                self.data[:, int(np.floor(it))] = add_weighted_avg(pre_trace, pre_weight, new_trace, 1.0 - portion)
                self.dp[int(np.floor(it))] += 1.0 - portion

                pro_trace = self.data[:, int(np.ceil(it))]
                pro_weight = self.dp[int(np.ceil(it))]
                self.data[:, int(np.ceil(it))] = add_weighted_avg(pro_trace, pro_weight, new_trace, portion)
                self.dp[int(np.ceil(it))] += portion

    def insert_trace_coordi(self, new_trace, coordi, max_distance=10.0):
        it, _, distance = self.coordi_to_idx(coordi)
        if distance < max_distance:
            self.insert_trace_idx(new_trace, it)
        else:
            print("Too far distance: ", distance)

    def insert_line(self, line):
        for it in range(line.shape[1]):
            coordi = line.idx_to_coordi(it)
            trace = line.get_trace_idx(it)
            self.insert_trace_coordi(trace, coordi)

    def save_data(self, data_pool, tag):
        data_root = os.path.join(get_project_root(), "data", data_pool)
        np.save(os.path.join(data_root, tag), self.data)
        with open(os.path.join(data_root, tag + ".nphead"), "w") as f:
            f.write("shape\t" + str(self.shape[0]) + "\t" + str(self.shape[1]) + "\n")
            f.write("amp\t" + str(self.amp) + "\n")
            f.write("dz\t" + str(self.dz) + "\n")
            f.write("\n" + "trace coordi" + "\n")
            for it in range(self.shape[1]):
                coordi = self.idx_to_coordi(it)
                f.write(str(it) + "\t" + str(coordi.cx) + "\t" + str(coordi.cy) + "\n")


# This subroutine must be modified according to the SEG-Y file format.
# Shell script in the data directory need to be executed before this subroutine to extract .head, .suhead, .su, and .bin files
def convert_volume_segy_to_numpy(data_pool="southsea", tag=None, is_align_coordi=False, rescale=1, sz=None, ez=None):
    data_root = os.path.join(get_project_root(), "data", data_pool)
    suhead_path = os.path.join(data_root, tag + ".suhead")
    header_path = os.path.join(data_root, tag + ".head")
    binary_path = os.path.join(data_root, tag + ".bin")

    # "amp" variable indicate the possibly largest amplitude value, need to be modified according to the SEG-Y file
    with open(header_path, encoding="latin-1") as f:
        for _ in range(12):
            f.readline()

        amp_line = f.readline()
        amp_min = float(re.findall(r"[-+]?(?:\d*\.*\d+)", amp_line.split()[4])[0])
        amp_max = float(re.findall(r"[-+]?(?:\d*\.*\d+)", amp_line.split()[6])[0])
        amp = max(-amp_min, amp_max)

        # Examples
        # amp = 1.0
        # amp = 32767

    with open(suhead_path, "rb") as suhead_file:
        suhead = suhead_file.read()

        fth = suhead[:240]
        lth = suhead[-240:]

        # The number of traces along inline and crossline is extracted from su header.
        fix = struct.unpack("i", fth[0: 4])[0]
        lix = struct.unpack("i", lth[0: 4])[0]
        fiy = struct.unpack("i", fth[4: 8])[0]
        liy = struct.unpack("i", lth[4: 8])[0]

        nz = struct.unpack("h", fth[114: 116])[0]
        nx = lix - fix + 1
        ny = liy - fiy + 1
        dz = struct.unpack("h", fth[116: 118])[0] / 1000000

        trace_coordi = np.zeros((nx, ny, 2), dtype=np.float64)
        if is_align_coordi:  # Find the closest quadrangle to the traces' coordinates
            b = np.zeros((2 * nx * ny, 1), dtype=np.float64)
            A = np.zeros((2 * nx * ny, 8), dtype=np.float64)
            for iy in range(ny):
                for ix in range(nx):
                    it = ix + nx * iy
                    th = suhead[240 * it: 240 * (it + 1)]
                    cx = struct.unpack("i", th[72: 76])[0]
                    cy = struct.unpack("i", th[76: 80])[0]

                    b[2 * it + 0, 0] = cx
                    A[2 * it + 0, 0] = (nx - 1 - ix) * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 0, 2] = (nx - 1 - ix) * iy / (nx - 1) / (ny - 1)
                    A[2 * it + 0, 4] = ix * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 0, 6] = ix * iy / (nx - 1) / (ny - 1)

                    b[2 * it + 1, 0] = cy
                    A[2 * it + 1, 1] = (nx - 1 - ix) * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 1, 3] = (nx - 1 - ix) * iy / (nx - 1) / (ny - 1)
                    A[2 * it + 1, 5] = ix * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 1, 7] = ix * iy / (nx - 1) / (ny - 1)
            At = np.transpose(A)
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(At, A)), At), b)
            corner_coordi = [
                [Coordinate(x[0], x[1]), Coordinate(x[2], x[3])],
                [Coordinate(x[4], x[5]), Coordinate(x[6], x[7])],
            ]
            for ix in range(nx):
                for iy in range(ny):
                    x_portion = ix / (nx - 1)
                    y_portion = iy / (ny - 1)
                    coordi = (
                        corner_coordi[0][0] * (1.0 - x_portion) * (1.0 - y_portion) +
                        corner_coordi[0][1] * (1.0 - x_portion) * y_portion +
                        corner_coordi[1][0] * x_portion * (1.0 - y_portion) +
                        corner_coordi[1][1] * x_portion * y_portion
                    )
                    trace_coordi[ix, iy] = [coordi.cx, coordi.cy]
        else:
            for ix in range(nx):
                for iy in range(ny):
                    it = ix + nx * iy
                    th = suhead[240 * it: 240 * (it + 1)]
                    cx = struct.unpack("i", th[72: 76])[0]
                    cy = struct.unpack("i", th[76: 80])[0]
                    trace_coordi[ix, iy] = [cx, cy]

    data = read_binary(binary_path, nz)
    data = data.reshape(nz, ny, nx)
    data = np.transpose(data, (0, 2, 1))
    sz = 0  if sz is None else sz
    ez = nz if ez is None else ez
    rsz = max(sz, 0)
    rez = min(ez, nz)
    data = data[rsz: rez]
    if rescale != 1:
        rescale = int(rescale)
        data = data[::rescale]
        dz = dz * rescale
    nz = data.shape[0]

    np.save(os.path.join(data_root, tag), data)
    with open(os.path.join(data_root, tag + ".nphead"), "w") as f:
        f.write("shape\t" + str(nz) + "\t" + str(nx) + "\t" + str(ny) + "\n")
        f.write("amp\t" + str(amp) + "\n")
        f.write("dz\t" + str(dz) + "\n")
        f.write("\n" + "trace coordi" + "\n")
        for ix in range(nx):
            for iy in range(ny):
                f.write(str(ix) + "\t" + str(iy) + "\t" + str(trace_coordi[ix, iy, 0]) + "\t" + str(trace_coordi[ix, iy, 1]) + "\n")


# This class assumes a 3D seismic volume is distributed as a quadrangle shape
class SeismicVolume:
    def __init__(self, data_pool="southsea", tag=None):
        if tag is None:
            self.amp = 1.0
            self.dz = 1.0
            self.clean_data(shape=(1, 1, 1))
            self.corner_coordi = [
                [Coordinate(0.0, 0.0), Coordinate(0.0, 0.0)],
                [Coordinate(0.0, 0.0), Coordinate(0.0, 0.0)],
            ]
        else:
            data_root = os.path.join(get_project_root(), "data", data_pool)
            head_path = os.path.join(data_root, tag + ".nphead")
            data_path = os.path.join(data_root, tag + ".npy")

            self.read_nphead(head_path)
            self.data = np.load(data_path)
            print(np.max(np.abs(self.data)))
            self.data = self.data / self.amp
            if self.shape != self.data.shape:
                print(self.shape, self.data.shape)
                print("Data shape has error!")
                exit(-1)
            self.dp = np.ones(self.shape[1:], dtype=np.float32)

    def read_nphead(self, header_path):
        with open(header_path, encoding="latin-1") as f:
            shape_line = f.readline()
            nz = int(shape_line.split()[1])
            nx = int(shape_line.split()[2])
            ny = int(shape_line.split()[3])
            self.shape = (nz, nx, ny)

            amp_line = f.readline()
            self.amp = float(amp_line.split()[1])

            dz_line = f.readline()
            self.dz = float(dz_line.split()[1])

            for _ in range(2):
                f.readline()

            b = np.zeros((2 * nx * ny, 1), dtype=np.float64)
            A = np.zeros((2 * nx * ny, 8), dtype=np.float64)
            for iy in range(ny):
                for ix in range(nx):
                    coordi_line = f.readline()
                    ix = int(coordi_line.split()[0])
                    iy = int(coordi_line.split()[1])
                    it = ix + nx * iy
                    cx = float(coordi_line.split()[2])
                    cy = float(coordi_line.split()[3])

                    b[2 * it + 0, 0] = cx
                    A[2 * it + 0, 0] = (nx - 1 - ix) * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 0, 2] = (nx - 1 - ix) * iy / (nx - 1) / (ny - 1)
                    A[2 * it + 0, 4] = ix * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 0, 6] = ix * iy / (nx - 1) / (ny - 1)

                    b[2 * it + 1, 0] = cy
                    A[2 * it + 1, 1] = (nx - 1 - ix) * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 1, 3] = (nx - 1 - ix) * iy / (nx - 1) / (ny - 1)
                    A[2 * it + 1, 5] = ix * (ny - 1 - iy) / (nx - 1) / (ny - 1)
                    A[2 * it + 1, 7] = ix * iy / (nx - 1) / (ny - 1)
            At = np.transpose(A)
            x = np.matmul(np.matmul(np.linalg.inv(np.matmul(At, A)), At), b)
            self.corner_coordi = [
                [Coordinate(x[0], x[1]), Coordinate(x[2], x[3])],
                [Coordinate(x[4], x[5]), Coordinate(x[6], x[7])],
            ]

            # plt.scatter(b[0::2], b[1::2], s=3, c="red")
            # xs = []
            # ys = []
            # for ix in range(nx):
            #     for iy in range(ny):
            #         coordi = self.idx_to_coordi(ix, iy)
            #         xs.append(coordi.cx)
            #         ys.append(coordi.cy)
            # plt.scatter(xs, ys, s=3, c="blue")
            # plt.show()

    def clean_data(self, shape=None):
        if shape is None:
            self.data[:, :, :] = 0.0
            self.dp = np.zeros(self.shape[1:], dtype=np.float32)
        else:
            self.data = np.zeros(shape, dtype=np.float32)
            self.shape = shape
            self.dp = np.zeros(self.shape[1:], dtype=np.float32)

    def set_amp(self, amp=None):
        self.amp = amp if amp is not None else np.max(np.abs(self.data))
        self.data = self.data / self.amp

    def set_coordi(self, sscoordi=None, secoordi=None, escoordi=None, eecoordi=None):
        self.corner_coordi[0][0] = sscoordi if sscoordi is not None else self.corner_coordi[0][0]
        self.corner_coordi[0][1] = secoordi if secoordi is not None else self.corner_coordi[0][1]
        self.corner_coordi[1][0] = escoordi if escoordi is not None else self.corner_coordi[1][0]
        self.corner_coordi[1][1] = eecoordi if eecoordi is not None else self.corner_coordi[1][1]

    def cut_zaxis(self, sz, ez):
        rsz = max(sz, 0)
        rez = min(ez, self.shape[0])

        self.data = self.data[rsz: rez]
        self.shape = self.data.shape

    def idx_to_coordi(self, ix, iy):
        x_portion = ix / (self.shape[1] - 1)
        y_portion = iy / (self.shape[2] - 1)
        out_coordi = (
            self.corner_coordi[0][0] * (1.0 - x_portion) * (1.0 - y_portion) +
            self.corner_coordi[0][1] * (1.0 - x_portion) * y_portion +
            self.corner_coordi[1][0] * x_portion * (1.0 - y_portion) +
            self.corner_coordi[1][1] * x_portion * y_portion
        )
        return out_coordi

    def coordi_to_idx(self, coordi):
        coeff_1x = self.corner_coordi[1][1] - self.corner_coordi[1][0] - self.corner_coordi[0][1] + self.corner_coordi[0][0]
        coeff_1c = self.corner_coordi[0][1] - self.corner_coordi[0][0]
        coeff_2x = self.corner_coordi[1][0] - self.corner_coordi[0][0]
        coeff_2c = self.corner_coordi[0][0] - coordi

        coeff_1x = Coordinate(-coeff_1x.cy, coeff_1x.cx)
        coeff_1c = Coordinate(-coeff_1c.cy, coeff_1c.cx)

        q2 = (coeff_1x * coeff_2x).sum()
        q1 = (coeff_1x * coeff_2c + coeff_1c * coeff_2x).sum()
        q0 = (coeff_1c * coeff_2c).sum()
        if q2 > 1e-5:
            sq_det = (q1 * q1 - 4 * q2 * q0)**0.5
            x_portion_1 = (-q1 + sq_det) / (2 * q2)
            x_portion_2 = (-q1 - sq_det) / (2 * q2)
            if abs(x_portion_1 - 0.5) < abs(x_portion_2 - 0.5):
                x_portion = x_portion_1
            else:
                x_portion = x_portion_2
        else:
            x_portion = -q0 / q1

        ix = (self.shape[1] - 1) * x_portion

        coeff_1x = self.corner_coordi[1][1] - self.corner_coordi[1][0] - self.corner_coordi[0][1] + self.corner_coordi[0][0]
        coeff_1c = self.corner_coordi[1][0] - self.corner_coordi[0][0]
        coeff_2x = self.corner_coordi[0][1] - self.corner_coordi[0][0]
        coeff_2c = self.corner_coordi[0][0] - coordi

        coeff_1x = Coordinate(-coeff_1x.cy, coeff_1x.cx)
        coeff_1c = Coordinate(-coeff_1c.cy, coeff_1c.cx)

        q2 = (coeff_1x * coeff_2x).sum()
        q1 = (coeff_1x * coeff_2c + coeff_1c * coeff_2x).sum()
        q0 = (coeff_1c * coeff_2c).sum()
        if q2 > 1e-5:
            sq_det = (q1 * q1 - 4 * q2 * q0)**0.5
            y_portion_1 = (-q1 + sq_det) / (2 * q2)
            y_portion_2 = (-q1 - sq_det) / (2 * q2)
            if abs(y_portion_1 - 0.5) < abs(y_portion_2 - 0.5):
                y_portion = y_portion_1
            else:
                y_portion = y_portion_2
        else:
            y_portion = -q0 / q1

        iy = (self.shape[2] - 1) * y_portion
        return ix, iy

    def get_trace_idx(self, ix, iy):
        if ix < 0 or ix > self.shape[1] - 1 or iy < 0 or iy > self.shape[2] - 1:
            trace = np.zeros(self.shape[0], dtype=np.float32)
        else:
            if abs(round(ix) - ix) < 0.01:
                if abs(round(iy) - iy) < 0.01:
                    trace = self.data[:, int(round(ix)), int(round(iy))]
                else:
                    portion = iy - np.floor(iy)
                    pre_trace = self.data[:, int(round(ix)), int(np.floor(iy))]
                    pro_trace = self.data[:, int(round(ix)), int(np.ceil(iy))]
                    trace = portion * pro_trace + (1.0 - portion) * pre_trace
            else:
                if abs(round(iy) - iy) < 0.01:
                    portion = ix - np.floor(ix)
                    pre_trace = self.data[:, int(np.floor(ix)), int(round(iy))]
                    pro_trace = self.data[:, int(np.ceil(ix)), int(round(iy))]
                    trace = portion * pro_trace + (1.0 - portion) * pre_trace
                else:
                    x_portion = ix - np.floor(ix)
                    y_portion = iy - np.floor(iy)
                    prepre_trace = self.data[:, int(np.floor(ix)), int(np.floor(iy))]
                    prepro_trace = self.data[:, int(np.floor(ix)), int(np.ceil(iy))]
                    propre_trace = self.data[:, int(np.ceil(ix)), int(np.floor(iy))]
                    propro_trace = self.data[:, int(np.ceil(ix)), int(np.ceil(iy))]
                    trace = (
                        x_portion * y_portion * propro_trace +
                        x_portion * (1.0 - y_portion) * propre_trace +
                        (1.0 - x_portion) * y_portion * prepro_trace +
                        (1.0 - x_portion) * (1.0 - y_portion) * prepre_trace
                    )
        return trace

    def get_trace_coordi(self, coordi, mode="bilinear"):
        ix, iy = self.coordi_to_idx(coordi)
        if mode == "nearest":
            ix = round(ix)
            iy = round(iy)
        return self.get_trace_idx(ix, iy)

    def insert_trace_idx(self, new_trace, ix, iy):
        assert len(new_trace) == self.shape[0]
        if 0 <= ix <= self.shape[1] - 1 and 0 <= iy <= self.shape[2] - 1:
            if abs(round(ix) - ix) < 0.01:
                if abs(round(iy) - iy) < 0.01:
                    trace = self.data[:, int(round(ix)), int(round(iy))]
                    weight = self.dp[int(round(ix)), int(round(iy))]
                    self.data[:, int(round(ix)), int(round(iy))] = add_weighted_avg(trace, weight, new_trace, 1.0)
                    self.dp[int(round(ix)), int(round(iy))] += 1.0
                else:
                    portion = iy - np.floor(iy)

                    pre_trace = self.data[:, int(round(ix)), int(np.floor(iy))]
                    pre_weight = self.dp[int(round(ix)), int(np.floor(iy))]
                    self.data[:, int(round(ix)), int(np.floor(iy))] = add_weighted_avg(pre_trace, pre_weight, new_trace, 1.0 - portion)
                    self.dp[int(round(ix)), int(np.floor(iy))] += 1.0 - portion

                    pro_trace = self.data[:, int(round(ix)), int(np.ceil(iy))]
                    pro_weight = self.dp[int(round(ix)), int(np.ceil(iy))]
                    self.data[:, int(round(ix)), int(np.ceil(iy))] = add_weighted_avg(pro_trace, pro_weight, new_trace, portion)
                    self.dp[int(round(ix)), int(np.ceil(iy))] += portion
            else:
                if abs(round(iy) - iy) < 0.01:
                    portion = ix - np.floor(ix)

                    pre_trace = self.data[:, int(np.floor(ix)), int(round(iy))]
                    pre_weight = self.dp[int(np.floor(ix)), int(round(iy))]
                    self.data[:, int(np.floor(ix)), int(round(iy))] = add_weighted_avg(pre_trace, pre_weight, new_trace, 1.0 - portion)
                    self.dp[int(np.floor(ix)), int(round(iy))] += 1.0 - portion

                    pro_trace = self.data[:, int(np.ceil(ix)), int(round(iy))]
                    pro_weight = self.dp[int(np.ceil(ix)), int(round(iy))]
                    self.data[:, int(np.ceil(ix)), int(round(iy))] = add_weighted_avg(pro_trace, pro_weight, new_trace, portion)
                    self.dp[int(np.ceil(ix)), int(round(iy))] += portion
                else:
                    x_portion = ix - np.floor(ix)
                    y_portion = iy - np.floor(iy)

                    prepre_trace = self.data[:, int(np.floor(ix)), int(np.floor(iy))]
                    prepre_weight = self.dp[int(np.floor(ix)), int(np.floor(iy))]
                    self.data[:, int(np.floor(ix)), int(np.floor(iy))] = add_weighted_avg(prepre_trace, prepre_weight, new_trace, (1.0 - x_portion) * (1.0 - y_portion))
                    self.dp[int(np.floor(ix)), int(np.floor(iy))] += (1.0 - x_portion) * (1.0 - y_portion)

                    prepro_trace = self.data[:, int(np.floor(ix)), int(np.ceil(iy))]
                    prepro_weight = self.dp[int(np.floor(ix)), int(np.ceil(iy))]
                    self.data[:, int(np.floor(ix)), int(np.ceil(iy))] = add_weighted_avg(prepro_trace, prepro_weight, new_trace, (1.0 - x_portion) * y_portion)
                    self.dp[int(np.floor(ix)), int(np.ceil(iy))] += (1.0 - x_portion) * y_portion

                    propre_trace = self.data[:, int(np.ceil(ix)), int(np.floor(iy))]
                    propre_weight = self.dp[int(np.ceil(ix)), int(np.floor(iy))]
                    self.data[:, int(np.ceil(ix)), int(np.floor(iy))] = add_weighted_avg(propre_trace, propre_weight, new_trace, x_portion * (1.0 - y_portion))
                    self.dp[int(np.ceil(ix)), int(np.floor(iy))] += x_portion * (1.0 - y_portion)

                    propro_trace = self.data[:, int(np.ceil(ix)), int(np.ceil(iy))]
                    propro_weight = self.dp[int(np.ceil(ix)), int(np.ceil(iy))]
                    self.data[:, int(np.ceil(ix)), int(np.ceil(iy))] = add_weighted_avg(propro_trace, propro_weight, new_trace, x_portion * y_portion)
                    self.dp[int(np.ceil(ix)), int(np.ceil(iy))] += x_portion * y_portion

    def get_dimension(self):
        xf = distance(self.corner_coordi[0][0], self.corner_coordi[1][0])
        xe = distance(self.corner_coordi[0][1], self.corner_coordi[1][1])

        yf = distance(self.corner_coordi[0][0], self.corner_coordi[0][1])
        ye = distance(self.corner_coordi[1][0], self.corner_coordi[1][1])

        return (xf, xe), (yf, ye)

    def check_coordi_inside(self, coordi):
        ix, iy = self.coordi_to_idx(coordi)
        return 0 <= ix <= self.shape[1] - 1 and 0 <= iy <= self.shape[2] - 1

    def insert_trace_coordi(self, new_trace, coordi):
        ix, iy = self.coordi_to_idx(coordi)
        self.insert_trace_idx(new_trace, ix, iy)

    def get_line(self, line, mode="bilinear"):
        for it in range(line.shape[1]):
            coordi = line.idx_to_coordi(it)
            trace = self.get_trace_coordi(coordi, mode=mode)
            line.insert_trace_idx(trace, it)
        return line

    def get_line_coordi(self, scoordi, ecoordi, nt, mode="bilinear"):
        line = SeismicLine(tag=None)
        line.clean_data(shape=(self.shape[0], nt))
        line.set_coordi(scoordi, ecoordi)
        return self.get_line(line, mode=mode)

    def get_line_idx(self, sx, sy, ex, ey, nt=None):
        nt = max(ex - sx, ey - sy) if nt is None else nt
        scoordi = self.idx_to_coordi(sx, sy)
        ecoordi = self.idx_to_coordi(ex, ey)
        return self.get_line_coordi(scoordi, ecoordi, nt)

    def insert_line(self, line):
        scoordi = line.scoordi
        ecoordi = line.ecoordi

        ineq00 = determinant(self.corner_coordi[0][0] - scoordi, ecoordi - scoordi) > 0
        ineq01 = determinant(self.corner_coordi[0][1] - scoordi, ecoordi - scoordi) > 0
        ineq10 = determinant(self.corner_coordi[1][0] - scoordi, ecoordi - scoordi) > 0
        ineq11 = determinant(self.corner_coordi[1][1] - scoordi, ecoordi - scoordi) > 0

        if (ineq00 and ineq01 and ineq10 and ineq11) or (not ineq00 and not ineq01 and not ineq10 and not ineq11):
            return

        for it in range(line.shape[1]):
            coordi = line.idx_to_coordi(it)
            trace = line.get_trace_idx(it)
            self.insert_trace_coordi(trace, coordi)

    def insert_volume(self, volume, is_force=False):
        conditions = (
            is_force
            or any(self.check_coordi_inside(c) for c in (
                volume.corner_coordi[0][0],
                volume.corner_coordi[0][1],
                volume.corner_coordi[1][0],
                volume.corner_coordi[1][1],
            ))
            or any(volume.check_coordi_inside(c) for c in (
                self.corner_coordi[0][0],
                self.corner_coordi[0][1],
                self.corner_coordi[1][0],
                self.corner_coordi[1][1],
            ))
        )
        if conditions:
            for ix in range(volume.shape[1]):
                for iy in range(volume.shape[2]):
                    coordi = volume.idx_to_coordi(ix, iy)
                    trace = volume.get_trace_idx(ix, iy)
                    self.insert_trace_coordi(trace, coordi)

    def save_data(self, data_pool, tag):
        data_root = os.path.join(get_project_root(), "data", data_pool)
        np.save(os.path.join(data_root, tag), self.data)
        with open(os.path.join(data_root, tag + ".nphead"), "w") as f:
            f.write("shape\t" + str(self.shape[0]) + "\t" + str(self.shape[1]) + "\t" + str(self.shape[2]) + "\n")
            f.write("amp\t" + str(self.amp) + "\n")
            f.write("dz\t" + str(self.dz) + "\n")
            f.write("\n" + "trace coordi" + "\n")
            for ix in range(self.shape[1]):
                for iy in range(self.shape[2]):
                    coordi = self.idx_to_coordi(ix, iy)
                    f.write(str(ix) + "\t" + str(iy) + "\t" + str(coordi.cx) + "\t" + str(coordi.cy) + "\n")


# Usage examples
if __name__ == "__main__":
    from utils.data import show_2d_array

    facies_data = np.load(os.path.join(get_project_root(), "data", "southsea", "3dn_facies_fillbot.npy"))
    print(facies_data.shape)
    nz, nx, ny = facies_data.shape

    new_facies_data = np.zeros_like(facies_data)
    new_facies_data[facies_data == 0] = 0
    new_facies_data[facies_data == 1] = 2
    new_facies_data[facies_data == 2] = 3
    new_facies_data[facies_data == 3] = 3
    new_facies_data[facies_data == 4] = 3
    new_facies_data[facies_data == 5] = 3
    new_facies_data[facies_data == 6] = 4
    new_facies_data[facies_data == 7] = 5

    nonzero = new_facies_data != 0
    has_nonzero = nonzero.any(axis=0)
    first_idx = np.argmax(nonzero, axis=0)
    z = np.arange(nz)[:, None, None]
    leading_top = (z < first_idx) & has_nonzero[None]
    new_facies_data[leading_top & (new_facies_data == 0)] = 1

    np.save(os.path.join(get_project_root(), "data", "southsea", "3dn_facies_newclass.npy"), new_facies_data)

    "show numpy data"
    for i in range(50, 2000, 100):
        img = new_facies_data[:, i]
        show_2d_array(img, scale=200, vmin=0, vmax=5, cmap="gray")
