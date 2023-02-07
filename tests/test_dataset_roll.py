""" Test the different methods of transforming the longitude coordinates after rolling"""

import numpy as np


# calculate the offset in the same way as in dataset_utils
def calculate_offset(a, bounds):
    low, high = bounds
    # get resolution of data
    res = a[1] - a[0]

    # calculate how many degrees to move by to have lon[0] of rolled subset as lower bound of request
    diff = a[0] - low

    # work out how many elements to roll by to roll data by 1 degree
    index = 1 / res

    # calculate the corresponding offset needed to change data by diff
    # round up to ensure rolling by enough
    # offset = math.ceil(diff * index)
    offset = int(round(diff * index))

    return offset


def dataset_roll(a, offset, bounds):
    # roll the dataset
    low, high = bounds
    a_roll = np.roll(a, offset)

    if offset < 0:
        a_new = np.where(
            np.logical_and(low >= a_roll, a_roll <= -(360 + low)), a_roll, a_roll % 360
        )  # this doesn't work in all negative offset cases
    else:
        a_new = np.where(a_roll < (360 + low), a_roll, a_roll % -360)

    return a_new


def dataset_roll_using_offset(a, offset):
    # roll the dataset
    a_roll = np.roll(a, offset)

    if offset < 0:
        a_roll[offset:] = a_roll[offset:] % 360
    else:
        a_roll[:offset] = a_roll[:offset] % -360

    return a_roll


class TestLonRoll_0_360:
    # offset = 359
    def test_to_minus_359_0(self):
        a = np.arange(start=0, stop=360, step=1)
        bounds = (-359, 0)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-359, stop=1, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-359, stop=1, step=1))

    # offset = 270
    def test_to_minus_270_to_89(self):
        a = np.arange(start=0, stop=360, step=1)
        bounds = (-270, 89)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-270, stop=90, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-270, stop=90, step=1))

    # offset = 180
    def test_to_minus_180_179(self):
        a = np.arange(start=0, stop=360, step=1)
        bounds = (-180, 179)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-180, stop=180, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-180, stop=180, step=1))

    # offset = 90
    def test_to_minus_90_269(self):
        a = np.arange(start=0, stop=360, step=1)
        bounds = (-90, 269)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-90, stop=270, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-90, stop=270, step=1))

    # offset = 0
    def test_to_0_359(self):
        a = np.arange(start=0, stop=360, step=1)
        bounds = (0, 359)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=0, stop=360, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=0, stop=360, step=1))


class TestLonRoll_minus_90_270:
    # offset = 269
    def test_to_minus_359_0(self):
        a = np.arange(start=-90, stop=270, step=1)
        bounds = (-359, 0)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-359, stop=1, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-359, stop=1, step=1))

    # offset = 180
    def test_to_minus_270_to_89(self):
        a = np.arange(start=-90, stop=270, step=1)
        bounds = (-270, 89)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-270, stop=90, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-270, stop=90, step=1))

    # offset = 90
    def test_to_minus_180_179(self):
        a = np.arange(start=-90, stop=270, step=1)
        bounds = (-180, 179)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-180, stop=180, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-180, stop=180, step=1))

    # offset = 0
    def test_to_minus_90_269(self):
        a = np.arange(start=-90, stop=270, step=1)
        bounds = (-90, 269)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-90, stop=270, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-90, stop=270, step=1))

    # offset = -90 - FAILS
    def test_to_0_359(self):
        a = np.arange(start=-90, stop=270, step=1)
        bounds = (0, 359)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=0, stop=360, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=0, stop=360, step=1))


class TestLonRoll_minus_180_180:
    # offset = 179
    def test_to_minus_359_0(self):
        a = np.arange(start=-180, stop=180, step=1)
        bounds = (-359, 0)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-359, stop=1, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-359, stop=1, step=1))

    # offset = 90
    def test_to_minus_270_to_89(self):
        a = np.arange(start=-180, stop=180, step=1)
        bounds = (-270, 89)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-270, stop=90, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-270, stop=90, step=1))

    # offset = 0
    def test_to_minus_180_179(self):
        a = np.arange(start=-180, stop=180, step=1)
        bounds = (-180, 179)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-180, stop=180, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-180, stop=180, step=1))

    # offset = -90 - FAILS
    def test_to_minus_90_269(self):
        a = np.arange(start=-180, stop=180, step=1)
        bounds = (-90, 269)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=-90, stop=270, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-90, stop=270, step=1))

    # offset = -180 - FAILS
    def test_to_0_359(self):
        a = np.arange(start=-180, stop=180, step=1)
        bounds = (0, 359)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=0, stop=360, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=0, stop=360, step=1))


class TestLonRoll_minus_270_90:
    # offset = 89
    def test_to_minus_359_0(self):
        a = np.arange(start=-270, stop=90, step=1)
        bounds = (-359, 0)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-359, stop=1, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-359, stop=1, step=1))

    # offset = 0
    def test_to_minus_270_to_89(self):
        a = np.arange(start=-270, stop=90, step=1)
        bounds = (-270, 89)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-270, stop=90, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-270, stop=90, step=1))

    # offset = -90 - fAILS
    def test_to_minus_180_179(self):
        a = np.arange(start=-270, stop=90, step=1)
        bounds = (-180, 179)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=-180, stop=180, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-180, stop=180, step=1))

    # offset = -180 - FAILS
    def test_to_minus_90_269(self):
        a = np.arange(start=-270, stop=90, step=1)
        bounds = (-90, 269)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=-90, stop=270, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-90, stop=270, step=1))

    # offset = -270 - FAILS
    def test_to_0_359(self):
        a = np.arange(start=-270, stop=90, step=1)
        bounds = (0, 359)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=0, stop=360, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=0, stop=360, step=1))


class TestLonRoll_minus_360_0:
    # offset = 0
    def test_to_minus_359_0(self):
        a = np.arange(start=-359, stop=1, step=1)
        bounds = (-359, 0)
        offset = calculate_offset(a, bounds)

        a_where = dataset_roll(a, offset, bounds)
        assert np.array_equal(a_where, np.arange(start=-359, stop=1, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-359, stop=1, step=1))

    # offset = -89
    def test_to_minus_270_to_89(self):
        a = np.arange(start=-359, stop=1, step=1)
        bounds = (-270, 89)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=-270, stop=90, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-270, stop=90, step=1))

    # offset = -179
    def test_to_minus_180_179(self):
        a = np.arange(start=-359, stop=1, step=1)
        bounds = (-180, 179)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=-180, stop=180, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-180, stop=180, step=1))

    # offset = -269
    def test_to_minus_90_269(self):
        a = np.arange(start=-359, stop=1, step=1)
        bounds = (-90, 269)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=-90, stop=270, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=-90, stop=270, step=1))

    # offset = -359
    def test_to_0_359(self):
        a = np.arange(start=-359, stop=1, step=1)
        bounds = (0, 359)
        offset = calculate_offset(a, bounds)

        # a_where = dataset_roll(a, offset, bounds)
        # assert np.array_equal(a_where, np.arange(start=0, stop=360, step=1))

        a_offset = dataset_roll_using_offset(a, offset)
        assert np.array_equal(a_offset, np.arange(start=0, stop=360, step=1))
