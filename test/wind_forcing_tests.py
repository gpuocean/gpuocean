# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2022 SINTEF Digital

This python program runs unit and integration tests related to  simply
importing packages of gpuocean.

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

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

from windForcing.WindForcing_Test import WindForcingTest

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 1):
    if (sys.argv[1].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))

   
loader = unittest.TestLoader()

suite = unittest.TestSuite([loader.loadTestsFromTestCase(WindForcingTest)])
results = unittest.TextTestRunner(verbosity=2).run(suite)

sys.exit(not results.wasSuccessful())