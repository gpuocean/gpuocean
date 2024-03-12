# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2017, 2018 SINTEF Digital

This python program runs unit and integration tests related to the 
stochastic components of GPU Ocean.

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
from stochastic.RandomNumbers_test import RandomNumbersTest
from stochastic.OceanStateNoise_test import OceanStateNoiseTest
from stochastic.OceanStateNoise_LCG_test import OceanStateNoiseLCGTest
from stochastic.RandThroughOceanNoise_test import RandThroughOceanNoiseTest
from stochastic.RandThroughOceanNoise_LCG_test import RandThroughOceanNoiseLCGTest

def printSupportedTests():
    print ("Supported tests:")
    print ("0: All, 1: RandomNumbers, " +
           "2: RandThroughOceanNoiseTest, 3: OceanStateNoise, " +
           "4: RandThroughOceanNoiseLCGTest, 5: OceanStateNoiseLCGTest")


if (len(sys.argv) < 2):
    print ("Usage:")
    print ("\t %s tests  [jenkins]" % sys.argv[0])
    printSupportedTests()
    exit()
tests = int(sys.argv[1])

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 2):
    if (sys.argv[1].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
if tests == 0:
    test_classes_to_run = [RandomNumbersTest,
                           RandThroughOceanNoiseTest, 
                           OceanStateNoiseTest,
                           RandThroughOceanNoiseLCGTest,
                           OceanStateNoiseLCGTest]
elif tests == 1:
    test_classes_to_run = [RandomNumbersTest]
elif tests == 2:
    test_classes_to_run = [RandThroughOceanNoiseTest]
elif tests == 3:
    test_classes_to_run = [OceanStateNoiseTest]
elif tests == 4:
    test_classes_to_run = [RandThroughOceanNoiseLCGTest]
elif tests == 5:
    test_classes_to_run = [OceanStateNoiseLCGTest]
else:
    print ("Error: " + str(tests) + " is not a supported test number...")
    printSupportedTests()
    exit()

    
loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
results = unittest.TextTestRunner(verbosity=2).run(big_suite)

sys.exit(not results.wasSuccessful())
