import numpy as np
from pathlib import Path
import pandas as pd
import uuid
from shutil import copyfile

inputDataStr = input("Select path where driving_log.csv + IMG folder are stored (absolute or relative). Output data will be copied to [SRC]/out/[CAMERA_PERSPECTIVE]): ")
inputDataPath = Path(inputDataStr.strip())
print('Searching driving_log.csv inside', inputDataPath)

prefix = input("Specify the prefix for the filenames: ")

# camera perspectives
perspectives = ['left', 'center', 'right']

# create output directories for each camera perspective
dest = {}
for p in perspectives:
	dest[p] = inputDataPath / f'out/{p}' # copy output data to subdirectory out/CAMERA_PERSPECTIVE
	dest[p].mkdir(parents=True, exist_ok=True) # create subdirectories, skip if already exists

df = pd.read_csv(
	inputDataPath / 'driving_log.csv', # source CSV file
	delimiter=',', # , as column seperation
	header=None, # no header row
	names=['center', 'left', 'right', 'angle', 'acc', 'brake', 'speed'], # column names
	dtype={'center': str, 'left': str, 'right': str, 'angle': float, 'acc': float, 'break': float, 'speed': float} # data types
)

print('Start copying frames into', inputDataStr, '/out/<CAMERA_PERSPECTIVE>/<Index>_<codedSpeed>_<codedAngle>.jpg')
copyCount = 0
for idx, row in df.iterrows():
	# create new filename: <Index>_<coded_speed>_<coded_angle>.jpg
	uuidVal = uuid.uuid1() # Generate a UUID from a host ID, sequence number, and the current time. --> for future use
	codedSpeed = int(row['speed'] * 1000)
	codedAngle = int(row['angle'] * 1000)
	fname = f'{prefix}{idx}_{codedSpeed}_{codedAngle}.jpg'
	#print(fname)

	srcLeft = Path(row['left'].strip())
	srcCenter = Path(row['center'].strip())
	srcRight = Path(row['right'].strip())

	# Try to copy source files into destination directories, re-labeled
	try:
		for p in perspectives:
			srcImage = Path(row[p].strip()) # The driving_log.csv file is storing absolute path names --> convert absolute path string to Path object of each source frame
			copyfile(srcImage, dest[p] / fname)
		copyCount += 1
	except Exception as e:
		print(e)
		continue

print(f'Successfully copied {copyCount}/{len(df)} frames.')
