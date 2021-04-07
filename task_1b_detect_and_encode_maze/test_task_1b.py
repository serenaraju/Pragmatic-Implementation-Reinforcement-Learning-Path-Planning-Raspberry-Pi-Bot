
'''
*****************************************************************************************
*
*        =================================================
*             Nirikshak Bot (NB) Theme (eYRC 2020-21)
*        =================================================
*
*  This script is intended to check the output of Task 1B
*  of Nirikshak Bot (NB) Theme (eYRC 2020-21).
*
*  Filename:			test_task_1b.py
*  Created:				11/10/2020
*  Last Modified:		12/10/2020
*  Author:				e-Yantra Team
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using ICT (NMEICT)
*
*****************************************************************************************
'''


# Import test module
try:
	import task_1b_cardinal

except ImportError:
	print("\n\t[ERROR] It seems that task_1b_cardinal.pyc is not found in current directory! OR")
	print("\n\tAlso, it might be that you are running test_task_1b.py from outside the Conda environment!\n")
	exit()


# Main function
if __name__ == '__main__':

	task_1b_cardinal.main()

