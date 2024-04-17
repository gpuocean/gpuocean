# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2017, 2018 SINTEF Digital

This python program runs unit test of some of the basic building 
blocks in the GPU Ocean software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import sys
import time

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

#import testUtils
from pycudaTests.CUDAArray2D_test import CUDAArray2DTest
from pycudaTests.CUDATexture_test import CUDACDKLMTextureTest

if (len(sys.argv) < 1):
    print("Usage:")
    print("\t %s  [jenkins]" % sys.argv[0])
    exit()

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 1):
    if (sys.argv[1].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
test_classes_to_run = [CUDAArray2DTest]


loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
results = unittest.TextTestRunner(verbosity=2).run(big_suite)

sys.exit(not results.wasSuccessful())
