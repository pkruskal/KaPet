from enum import Enum

# standardizing column names to a common set
class ColumnNames(Enum):
	"""
	label : (float) column name for the the targrets we are training on
	image_path: (str) column name for the path to where the image lives
	image_name: (str) identification string for the image
	"""
	label = "Pawpularity"
	image_path = "image_path"
	image_name = "Id"

class DefaultFeatureColumnNames(Enum):
	"""
	Columns names for the provided default features
	"""
	Subject = 'Subject Focus'
	Eyes = 'Eyes'
	Face = 'Face'
	Near = 'Near'
	Action = 'Action'
	Accessory = 'Accessory'
	Group = 'Group'
	Collage = 'Collage'
	Human = 'Human'
	Occlusion = 'Occlusion'
	Info = 'Info'
	Blur = 'Blur'
