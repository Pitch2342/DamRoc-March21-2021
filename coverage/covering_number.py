# import pybloomfilter
import numpy
import random

class Coverage(object):

    def __init__(self, num_samples, error_rate=0.000001):
        self._num_samples = num_samples
        self._expected_num_points = num_samples
        self._denominator = self._expected_num_points
        self._numerator = 0.0

#         self._bf = pybloomfilter.BloomFilter(self._expected_num_points, error_rate)
        self._bf = []

        print("\n\n======================\n===================")
        print("INIT START")
        print("=============")
        print("_num_samples")
        print(self._num_samples)
        print("=============")
        print("_expected_num_points")
        print(self._expected_num_points)
        print("=============")
        print("INIT END\n\n======================\n===================")

    def update(self, point):

#         print("\n\n======================\n===================")
#         print("UPDATE NUMERATOR START")
#         print("=============")
#         print("point")
#         print(point)
#         print("=============")
        pt_np = point.numpy().tolist()
#         is_not_in = not self._bf.add(hash(str(pt_np)))
#         self._numerator += int(is_not_in)
        if str(pt_np) not in self._bf:
            self._numerator += 1
            self._bf.append(str(pt_np))
            print("NEW POINT")
        print("=============")
#         print("is_not_in")
#         print(is_not_in)
#         print("=============")
#         print("self._numerator")
#         print(self._numerator)
#         print("=============")
#         print("UPDATE NUMERATOR END\n\n======================\n===================")
        return(self._numerator/(self._num_samples))
        
    def Covering_value(self):
        """
        :return: the ratio of the visited samples to the maximum expected ones
        """
        print("\n\n======================\n===================")
        print("RATIO START")
        print("=============")
        print("self._actual_num_points")
        print(self._numerator)
        print("=============")
        print("self._expected_num_points")
        print(self._denominator)
        print("=============")
        print("self._actual_num_points * 1. / self._expected_num_points")
        print(self._numerator * 1. / self._denominator)
        print("=============")
        print("RATIO END\n\n======================\n===================")
        return self._numerator * 1. / self._denominator