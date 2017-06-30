import os
import shutil
import pandas as pd
import zipfile
import sys
import tarfile
import cv2
import errno
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-l", "--library", dest="library", help="Name of the library used (mxnet or keras")
parser.add_option("-i", "--input-folder", dest="input_folder", help="Data input folder")
parser.add_option("-t", "--task", dest="task", help="Task type (table or value)")
(options, args) = parser.parse_args()


input_path = os.path.join('/home/ubuntu/workspace/keras-frcnn/data', options.input_folder)
outdir = os.path.join('/home/ubuntu/workspace/keras-frcnn/data', options.input_folder, 'Out')

def extract_folders(indir):
	for root, dirs, filenames in os.walk(indir):
		#print(root)
		#print(dirs)
		for f in filenames:
			#print(f)
			if f.endswith('.zip'):
				try:
					tar = tarfile.open(indir + str(f))
					tar.extractall(path=outdir)
					tar.close()
				except OSError as oserr:
					if oserr.errno != errno.ENAMETOOLONG:
						continue# ignore
					else:
						print('File Name too long! - ' + str(f))


def rename_filenames(outdir):
	file_list = []

	for path, subdirs, files in os.walk(outdir):
		#print('Path: ' + str(path))
		#print('Subdirs: ' + str(subdirs))
		#print('Files: ' + str(files))
		tmp_file_list = []
		for name in files:
			#print(name)
			if name.endswith('.pdf'):
				pdf_name, file_ending = name.split('.pdf')
			elif name.endswith('.PDF'):
				pdf_name, file_ending = name.split('.PDF')
			if name.endswith('.json'):
				json_name = name
			if name.endswith('.jpeg'):
				normalized_name = name.replace('.jpeg', '.jpg')
				os.rename(os.path.join(path,name), os.path.join(path, normalized_name))
				name = normalized_name
				tmp_file_list.append(name)
		tmp_file_list.append(path)
		if 'json_name' in locals():
			tmp_file_list.append(json_name)
		if 'pdf_name' in locals():
			tmp_file_list.append(pdf_name)
		if len(tmp_file_list) > 1:
			file_list.append(tmp_file_list)
	return file_list


def create_xml(df):
	import xml.etree.ElementTree as ET

	for index, row in df.iterrows():
		#print(index, row)
		for item in row:
			#print(item)
			if isinstance(item, str):
				#print("File Name: " + item)
				file_name = item
				output_filename = file_name.replace(".jpg", ".xml")

				annotation = ET.Element("annotation")
				ET.SubElement(annotation, "folder").text = "VOC2007"
				ET.SubElement(annotation, "filename").text = file_name

				source = ET.SubElement(annotation, "source")
				ET.SubElement(source, "database").text = "The VOC2007 Database"
				ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
				ET.SubElement(source, "image").text = "flickr"
				ET.SubElement(source, "flickrid").text = "341012865"

				owner = ET.SubElement(annotation, "owner")
				ET.SubElement(owner, "flickrid").text = "SCDM"
				ET.SubElement(owner, "name").text = "SCDM Training Data"

			if isinstance(item, list):
				#print("List Name: " + str(item))
				if len(item) == 5:
					x_1 = item[0]
					y_1 = item[1]
					x_2 = item[2]
					y_2 = item[3]
					class_name = item[4]

					#size = ET.SubElement(annotation, "size")
					#ET.SubElement(size, "width").text = "353"
					#ET.SubElement(size, "height").text = "500"
					#ET.SubElement(size, "depth").text = "3"

					ET.SubElement(annotation, "segmented").text = "0"

					img_object = ET.SubElement(annotation, "object")
					ET.SubElement(img_object, "name").text = class_name
					#ET.SubElement(img_object, "pose").text = "Left"
					#ET.SubElement(img_object, "truncated").text = "1"
					ET.SubElement(img_object, "difficult").text = "0"
					bndbox = ET.SubElement(img_object, "bndbox")
					ET.SubElement(bndbox, "xmin").text = str(x_1)
					ET.SubElement(bndbox, "ymin").text = str(y_1)
					ET.SubElement(bndbox, "xmax").text = str(x_2)
					ET.SubElement(bndbox, "ymax").text = str(y_2)
		size = ET.SubElement(annotation, "size")
		ET.SubElement(size, "width").text = str(row[-1])
		ET.SubElement(size, "height").text = str(row[-2])
		#ET.SubElement(size, "depth").text = "3"

		tree = ET.ElementTree(annotation)
		tree.write(os.path.join(input_path, "Annotations", output_filename), xml_declaration=True, encoding='utf-8')


def create_txt(df):
	''' 
	filepath,x1,y1,x2,y2,class_name
	/data/imgs/img_001.jpg,837,346,981,456,cow
	'''
	with open('./data/' + options.input_folder + '/' + 'training_data.txt', 'a') as the_file:
		for index, row in df.iterrows():
			print('!!! NEW ROW !!!')
			print(index, row)
			for item in row:
				print(item)
				if isinstance(item, str):
					print("File Name: " + item)
					file_name = input_path + '/train_data/' + item
				if isinstance(item, list):
					#print("List Name: " + str(item))
					if len(item) == 5:
						x_1 = item[0]
						y_1 = item[1]
						x_2 = item[2]
						y_2 = item[3]
						class_name = item[4]
						print(file_name + ',' + str(x_1) + ',' + str(y_1) + ',' + str(x_2) + ',' + str(y_2) + ',' + class_name + '\n')
						the_file.write(file_name + ',' + str(x_1) + ',' + str(y_1) + ',' + str(x_2) + ',' + str(y_2) + ',' + class_name + '\n')


def read_json(json_file):
	import json
	from pprint import pprint

	with open(json_file) as data_file:
		data = json.load(data_file)
	#pprint(data)
	found_valid_value = False
	correct_class = False
	dict = {}

	if 'pdf' in data:
		pdf_name = data.get("pdf")
		print('PDF-NAME: ' + str(pdf_name))

	if 'tags' in data:
		#print('TAGS FOUND!!!')
		#print(data.get("tags"))
		tag_list = data.get("tags")
		for entry in tag_list:
			#print("Entry: " + str(entry))
			for sub_item in entry:
				#print(sub_item)
				if 'name' in sub_item:
					name = entry.get("name").lower()
					#print(name)
				elif 'fileName' in sub_item:
					file_name = entry.get("fileName")
					if file_name.endswith('.jpeg'):
						file_name = file_name.replace('.jpeg', '.jpg')
					found_valid_value = True
				elif 'x' in sub_item:
					x_1 = entry.get("x")
				elif 'y' in sub_item:
					y_1 = entry.get("y")
				elif 'width' in sub_item:
					width = entry.get("width")
				elif 'height' in sub_item:
					height = entry.get("height")
			if found_valid_value == True:
				if options.task == 'value':
					if 'mk-' in name:
						correct_class = True
				elif options.task == 'table':
					if 'column' in name or 'row' in name or 'header' in name or 'table' in name:
						correct_class = True
			if correct_class == True:
				if pdf_name.endswith('.pdf'):
					pdf_name = pdf_name.replace('.pdf','')
				elif pdf_name.endswith('.PDF'):
					pdf_name = pdf_name.replace('.PDF','')
				key = pdf_name + '_' + file_name + '_' + name
				dict[key] = [x_1, y_1, width, height, name]
				found_valid_value = False
				correct_class = False
	#print(dict)
	return dict


def create_dataset():
	"""
    extract all raw zip files, create train/test set and annotation files
    :param input_path: The path to the zip folders
    :return: None
    """
	print('***** PART 1 - UNZIPPING FILES *****')
	print(options.library)
	# 1 Unzip folder
	zip_path = os.path.join(input_path, 'Raw/')
	assert os.path.exists(zip_path), 'Path does not exist: {}'.format(zip_path)
	extract_folders(zip_path)

	# 2 Get the name from the pdf file and images
	print('***** PART 2 - GET FILE NAMES *****')
	print('Output directory: ' + str(outdir))
	file_list = rename_filenames(outdir)
	print('File list: ' + str(file_list))

	# 3 With 2 + 3 - Create: trainval.txt and test.txt
	print('***** PART 3 - CREATE FILES *****')
	renamed_file_list = []
	json_dictionary_list = []

	for file in file_list:
		print('*********** NEXT FILE **********')
		print(file)
		pdf_name = file[-1]
		json_name = file[-2]
		path = file[-3]

		for item in file[:-3]:
			#print(item)
			os.rename(os.path.join(path,item), os.path.join(path, item.replace(item, pdf_name + '_' + item)))
			renamed_file_list.append(pdf_name + '_' + item)
			# Move the file to the JPEGImages folder
			os.rename(os.path.join(path, pdf_name + '_' + item), os.path.join(input_path, "train_data", pdf_name + '_' + item))
		# Read the json file
		json_dictionary = read_json(os.path.join(path, json_name))
		json_dictionary_list.append(json_dictionary)

	print('********* FILE LIST *********')
	print(renamed_file_list)
	print(json_dictionary_list)

	# Merge the file and json list
	final_list = []
	for file in renamed_file_list:
		#print(file)
		tmp_final_list = []
		tmp_final_list.append(file)
		for item in json_dictionary_list:
			for key, value in item.items():
				#print(key)
				#print(value)
				if file in key:
					#print('KEY FOUND!!!' + str(key))
					tmp_final_list.append(value)
					#break
		final_list.append(tmp_final_list)
	print('********* MERGED FILE LIST *********')
	print(final_list)

	# Convert list into pandas dataframe
	df = pd.DataFrame(final_list)
	#	print(df)

	# Iterate over DF and change % to pixel and normalize classes
	size_list = []
	for index, row in df.iterrows():
		#print(index, row)
		for column_index, item in enumerate(row):
			#print(item)
			#print(type(item))
			#print('*****')
			#print(df.iloc[index, column_index])
			#print('*****')
			if isinstance(item, str):
				print(item)                
				size = cv2.imread(os.path.join(input_path, "train_data", item)).shape
				height = size[0]
				width = size[1]
				tmp_size_list = []
				tmp_size_list.append(height)
				tmp_size_list.append(width)
				size_list.append(tmp_size_list)
				#print(height, width)
			elif isinstance(item, list):
				print('List item: ' + str(item))
				# Check if the length is as expected
				if len(item) == 5:
					#print(item[0])
					if isinstance(item[0], unicode):
						print(type(item[0]))
						item[0] = int(float(item[0].replace("%", ""))*0.01 * width)
						#print(item[0])
						#print(item[1])
						item[1] = int(float(item[1].replace("%", ""))*0.01 * height)
						#print(item[1])
 						#print(item[2])
						item[2] = int(item[0] + (float(item[2].replace("%", ""))*0.01 * width))
						if item[2] > width:
							item[2] = width
						item[3] = int(item[1] + (float(item[3].replace("%", ""))*0.01 * height))
					#print(item[3])
					#print(item[4])
					# Normalize the classes
					if 'table' in item[4]:
						item[4] = 'table'
					elif 'column' in item[4] or 'colunm' in item[4]:
						item[4] = 'column'
					elif 'row' in item[4]:
						item[4] = 'row'
					elif 'header' in item[4]:
						item[4] = 'header'
					elif "_" in item[4]:
						item[4] = item[4].split("_", 1)[0]
                        
					tmp_position_list = []
					tmp_position_list.append(item[0])
					tmp_position_list.append(item[1])
					tmp_position_list.append(item[2])
					tmp_position_list.append(item[3])
					tmp_position_list.append(item[4])
					#print(tmp_position_list)
					# Update the data frame
					df.iloc[index, column_index] = tmp_position_list
				else:
					print("ERROR!!!")

	print('********* DATA FRAME *********')
	print(df)
	#df.to_pickle('meta_dataframe.csv')
	print('Size list: ' + str(size_list))
	print('********* IMAGE SIZE DATA FRAME *********')
	df_size = pd.DataFrame(size_list, columns=['Height', 'Width'])
	print(df_size)
	df_merged = pd.concat([df, df_size], axis=1)
	print('********* MERGED DATA FRAME *********')
	print(df_merged)
	print(df_merged.shape)

	# 5) Create the .xml file for every image
	if options.library == 'mxnet':
		print('***** PART 5 - CREATE XML FILES *****')
		create_xml(df_merged)
	elif options.library == 'keras':
		print('***** PART 5 - CREATE TXT FILES *****')
		create_txt(df_merged)
	else:
		print('!!!!! NO CORRECT LIBRARY USED !!!!!')

	# 6) Create training and test data set
	# Randomly sample 60% of your dataframe
	print('***** PART 6 - DATASET GENERATION *****')
	#print(df.iloc[:,0])
	#print(df[0])
	#new_df = df[0].apply(lambda x: x.replace(".jpg", ""))
	#new_df.to_csv(os.path.join(input_path,'ImageSets','Main','dataframe.csv'))

	#df_trainval = new_df.sample(frac=0.6)
	#df_test = new_df.loc[~new_df.index.isin(df_trainval.index)]
	#print("Training Data shape: " + str(df_trainval.shape))
	#print("Test Data shape: " + str(df_test.shape))

	# Write training and test files out
	#df_trainval.to_csv(os.path.join(input_path,'ImageSets','Main','trainval.txt'), header=None, index=None, sep=' ', mode='a')
	#df_test.to_csv(os.path.join(input_path,'ImageSets','Main','test.txt'), header=None, index=None, sep=' ', mode='a')
	print('***** PART 7 - CLEANUP *****')
	shutil.rmtree(os.path.join(input_path, 'Out'))
	print('***** SCRIPT FINISHED *****')

def main():
	create_dataset()


if __name__ == '__main__':
	main()