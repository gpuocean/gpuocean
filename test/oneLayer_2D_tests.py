# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2017, 2018 SINTEF Digital

This python program runs integration tests for the different numerical
schemes.

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
from schemes.FBL_test import FBLtest
from schemes.CTCS_test import CTCStest
from schemes.CDKLM16_test import CDKLM16test
from schemes.KP07_test import KP07test
from schemes.NetCDF_test import NetCDFtest
from schemes.NetCDF_test import NetCDFtest
from schemes.ConservationOfMass_test import ConservationOfMassTest
from schemes.CombinedCDKLM16_test import CombinedCDKLM16test
from schemes.RealisticForcing_Test import RealisticForcingTest

def printSupportedSchemes():
    print("Supported schemes:")
    print("0: All, 1: FBL, 2: CTCS, 3: CDKLM16, 4: KP07, 5: NetCDF interface, 6: Conservation of mass, 7: Combined CDKLM time stepping, 8: Realistic forcing")
    

if (len(sys.argv) < 2):
    print("Usage:")
    print("\t %s scheme [jenkins]" % sys.argv[0])
    printSupportedSchemes()
    sys.exit()
scheme = int(sys.argv[1])

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 2):
    if (sys.argv[2].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
if scheme == 0:
    test_classes_to_run = [FBLtest, CTCStest, CDKLM16test, KP07test, NetCDFtest, ConservationOfMassTest, CombinedCDKLM16test, RealisticForcingTest]
elif scheme == 1:
    test_classes_to_run = [FBLtest]
elif scheme == 2:
    test_classes_to_run = [CTCStest]
elif scheme == 3:
    test_classes_to_run = [CDKLM16test]
elif scheme == 4:
    test_classes_to_run = [KP07test]
elif scheme == 5:
    test_classes_to_run = [NetCDFtest]
elif scheme == 6:
    test_classes_to_run = [ConservationOfMassTest]
elif scheme == 7:
    test_classes_to_run = [CombinedCDKLM16test]
elif scheme == 8:
    test_classes_to_run = [RealisticForcingTest]
else:
    print("Error: " + str(scheme) + " is not a supported scheme...")
    printSupportedSchemes()
    exit()


loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
results = unittest.TextTestRunner(verbosity=2).run(big_suite)

sys.exit(not results.wasSuccessful())
