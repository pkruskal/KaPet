from enum import Enum

# standardizing column names to a common set
class ColumnNames(Enum):
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
