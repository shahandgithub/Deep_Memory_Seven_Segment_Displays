# Tests of code provided by the instructor
import numpy as np


def test0():
    bf_a = BooleanFunc()
    bf_b = BooleanFunc(13)
    bf_c = BooleanFunc(100, eval_return_if_error=True)
    bf_d = BooleanFunc(21, defining_list=21 * [3.1])
    bf_e = BooleanFunc(2, eval_return_if_error=True, defining_list=[False, True])

    print("{}\n{}\n{}\n{}\n{}".format(bf_a, bf_b, bf_c, bf_d, bf_e))

    bf_and = BooleanFunc(defining_list=[0, 0, 0, 1])
    print("the AND gate using ints:\n", bf_and, "\n")

    bf_and = BooleanFunc(defining_list=[False, False, False, True])
    print("the AND gate using bools:\n", bf_and, "\n")

    try:
        bf_bad = BooleanFunc(21, defining_list=1.11)
    except Exception as err:
        print(type(err), ":", err, "\n")

    try:
        bf_bad = BooleanFunc(21, defining_list=[False, True])
    except Exception as err:
        print(type(err), ":", err, "\n")

    try:
        bf_bad = BooleanFunc(6, defining_list="hi mom")
        print(bf_bad)
    except Exception as err:
        print(type(err), ":", err, "\n")


# Act 1 test
def test1():
    bf_and = BooleanFunc(defining_list=[False, False, False, True])
    even_func_w_errs_true = [0, "bad", 2, 4, 6, 8, 10, 12, 3.14, 14]
    greater_9_func_true = [10, 11, 12, 13, 14, 15]
    greater_3_func_false = [0, 1, 2, 3]

    bf_a = BooleanFunc(10)
    bf_b = BooleanFunc(16)
    bf_c = BooleanFunc(16)

    print("--- Testing constructors and mutators of AND, even, >9 and >3 ---")
    bf_a.set_truth_table_using(True, even_func_w_errs_true)
    bf_b.set_truth_table_using(True, greater_9_func_true)
    bf_c.set_truth_table_using(False, greater_3_func_false)

    for func in [bf_and, bf_a, bf_b, bf_c]:
        print(func)

    print("--- Testing intputs that cover the allowable and illegal values for AND ---")
    for input_x in range(10):
        print(bf_and.eval(input_x))
        print("AND({}) = {}".format(input_x, bf_and.get_state()))

    # You should test bf_a, bf_b, and bf_c as well


# Act 2 test
def test2():
    my_12_seg = MultiSegmentLogic(12)

    print("As constructed -------------------")
    print(my_12_seg)

    try:
        my_12_seg.eval(1)
    except AttributeError as err:
        print("\nExpected ... " + str(err) + "\n")

    for k in range(12):
        my_12_seg.set_segment(k, BooleanFunc(
            defining_list=[True, False, True, False]))

    print(my_12_seg)

    print("Evaluating my_12_seg at 2 (which should be True) -----------")
    my_12_seg.eval(2)
    print(my_12_seg.get_val_of_seg(2))
    print()
    print("segs 3, 5 and, illegal, 29:   ",
          str(my_12_seg.get_val_of_seg(3)),
          str(my_12_seg.get_val_of_seg(5)),
          str(my_12_seg.get_val_of_seg(29))
          )


# Act 3 test
def test3():
    my_7_seg = SevenSegmentLogic()

    print("As constructed -------------------")
    print(my_7_seg)

    try:
        my_7_seg.set_num_segs(8)  # should "throw"
    except ValueError as err:
        print("\nExpected ... " + str(err) + "\n")

    try:
        my_7_seg.eval(1)
    except AttributeError as err:
        print("\nNot Expected... " + str(err) + "\n")

    for input_x in range(16):
        my_7_seg.eval(input_x)
        print("\n| ", end='')
        for k in range(7):
            print(str(my_7_seg.get_val_of_seg(k)) + " | ", end='')
        print()


import numpy as np


class BooleanFunc:
    # static members and intended constants
    MAX_TABLE_SIZE = 65536  # that's 16 binary inputs
    MIN_TABLE_SIZE = 2  # that's 1 binary input
    DEFAULT_TABLE_SIZE = 4
    DEFAULT_FUNC = DEFAULT_TABLE_SIZE * [False]

    # initializer ("constructor") method -------------------
    def __init__(self, table_size=None, defining_list=None, eval_return_if_error=False):
        self.truth_table = None
        self.eval_return_if_error = eval_return_if_error
        self.state = eval_return_if_error
        self.counter = 0
        if table_size != None:
            self.table_size = table_size
        else:
            self.table_size = 0
        if not table_size and not defining_list:
            # passed neither list nor size
            table_size = self.DEFAULT_TABLE_SIZE
            defining_list = self.DEFAULT_FUNC
        elif table_size and not defining_list:
            # passed size but no list
            self.valid_table_size(table_size)  # raises, no return
            defining_list = table_size * [False]
        elif not table_size:
            # passed list but no size
            self.valid_defining_list(defining_list)  # raises, no return
            table_size = len(defining_list)
        else:
            if type(defining_list) == list:
                if len(defining_list) != table_size:
                    return None
                else:
                    self.truth_table = defining_list.copy()
                # passed both list and size
                self.valid_defining_list(defining_list)
                if len(defining_list) != table_size:
                    raise ValueError("Table size does not match list length in constructor.")
            else:
                return None

        # sanitize bools (e.g. (1.32, "hi", -99)->True,
        # (0.0, "", 0)->False)
        eval_return_if_error = bool(eval_return_if_error)
        defining_list = [bool(item) for item in defining_list]

        # assign instance members
        self.eval_return_if_error = eval_return_if_error
        self.state = eval_return_if_error
        self.table_size = table_size
        self.truth_table = np.array(defining_list, dtype=bool)

    # stringizer  -------------------------------
    def __str__(self):
        ret_str = "truth_table: " + str(self.truth_table) \
                  + "\nsize = " + str(self.table_size) \
                  + "\nerror return = " + str(self.eval_return_if_error) \
                  + "\ncurrent state = " + str(self.state) + "\n"
        return ret_str

    def set_truth_table_using(self, rarer_value, rarer_val_lst):
        k = 0
        while k < self.table_size:
            if self.truth_table[k] in rarer_val_lst:
                if rarer_value == True:
                    self.truth_table[k] = True
                else:
                    self.truth_table[k] = False
            else:
                if rarer_value == True:
                    self.truth_table[k] = False
                else:
                    self.truth_table[k] = True
            k += 1
        count = 0
        print('t_table_using:', self.truth_table)
        for each in rarer_val_lst:
            try:
                if each < 0 or each > self.table_size:
                    count += 1
            except:
                count = count
        if (count == len(rarer_val_lst)):
            return True

        return False

    def eval(self, input):
        if input > 0 and input < self.table_size:
            self.state = input
            return True
        else:
            self.eval_return_if_error = self.state
            return False

    def get_state(self):
        return self.state

    @classmethod
    def valid_table_size(cls, size):
        if not isinstance(size, int):
            raise TypeError("Table size must be an int.")
        if not (cls.MIN_TABLE_SIZE <= size <= cls.MAX_TABLE_SIZE):
            raise ValueError("Bad table size passed to constructor (legal range: {}-{}).".
                             format(cls.MIN_TABLE_SIZE, cls.MAX_TABLE_SIZE))

    @classmethod
    def valid_defining_list(cls, the_list):
        if not isinstance(the_list, list):
            raise ValueError("Bad type in constructor. defining_list must be type list.")
        if not (cls.MIN_TABLE_SIZE <= len(the_list) <= cls.MAX_TABLE_SIZE):
            raise ValueError("Bad list passed to constructor (its length is outside legal range: {}-{}).".
                             format(cls.MIN_TABLE_SIZE, cls.MAX_TABLE_SIZE))


class MultiSegmentLogic:
    MAX_SEGS = 1000
    MIN_SEGS = 1
    DEFAULT_SEGS = 7

    def __init__(self, num_segs=DEFAULT_SEGS):
        self.segs = [BooleanFunc()] * num_segs
        self.num_segs = num_segs
        self.input = 0

    def set_num_segs(self, num_segs):
        self.segs = [BooleanFunc()] * num_segs
        self.num_segs = num_segs

    def set_segment(self, seg_num, func_for_this_seg):
        try:
            self.segs[seg_num] = func_for_this_seg
            return True
        except:
            return False

    def eval(self, input):
        # for each in self.segs:
        #    each.eval(input)
        print('input', input)
        self.input = input
        return True

    def get_val_of_seg(self, num):
        try:
            print('num:', num)
            print('ssn:', self.segs[num])
            if num == 'a' or num == 'A':
                self.segs[10]
            elif num == 'b' or num == 'B':
                self.segs[11]
            elif num == 'c' or num == 'C':
                self.segs[12]
            elif num == 'd' or num == 'D':
                self.segs[13]
            elif num == 'e' or num == 'E':
                self.segs[14]
            elif num == 'f' or num == 'F':
                self.segs[15]
            return self.segs[num]
        except:
            return False


class SevenSegmentLogic(MultiSegmentLogic):

    def __init__(self):
        self.segs = []
        self.helper_segs()
        self.set_num_segs(7)
        self.load_all_funcs_long_way()

    def set_segment(self, k, func_for_this_seg):
        try:
            self.segs[k] = func_for_this_seg
            return True
        except:
            return False

    def load_all_funcs_long_way(self):
        k = 0
        while k <= 15:
            if k == 0:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, True, True, True, True, False]))
            elif k == 1:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[False, True, True, False, False, False, False]))
            elif k == 2:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, False, True, True, False, True]))
            elif k == 3:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, True, True, False, False, True]))
            elif k == 4:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[False, True, True, False, False, True, True]))
            elif k == 5:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, False, True, True, False, True, True]))
            elif k == 6:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, False, True, True, True, True, True]))
            elif k == 7:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, True, False, False, False, False]))
            elif k == 8:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, True, True, True, True, True]))
            elif k == 9:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, True, True, False, True, True]))
            elif k == 10:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, True, True, False, True, True, True]))
            elif k == 11:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[False, False, True, True, True, True, True]))
            elif k == 12:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, False, False, True, True, True, False]))
            elif k == 13:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[False, True, True, True, True, False, True]))
            elif k == 14:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, False, False, True, True, True, True]))
            elif k == 15:
                self.segs.append(0)
                self.set_segment(k, BooleanFunc(defining_list=[True, False, False, False, True, True, True]))
            print('kk', k)
            k += 1

    def set_num_segs(self, num_segs):
        print('t_segs:', num_segs)
        if num_segs != 7:
            raise ValueError('A very specific bad thing happened.')
        else:
            self.num_segs = num_segs

    def get_val_of_seg(self, k):
        print('self.input:', self.input)
        return self.segs[self.input].truth_table[k]

    def helper_segs(self):
        self.set_num_segs(7)


if __name__ == '__main__':
    test0()
    # TODO these won't work yet until they are implemented
    test1()
    test2()
    test3()
