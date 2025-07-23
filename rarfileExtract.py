import rarfile

# Extract all files
with rarfile.RarFile('/Users/alexpoulin/Downloads/ACW_v2-6.rar') as rf:
    rf.extractall('/Users/alexpoulin/Library/Application Support/Steam/steamapps/common/Napoleon Total War/NapoleonData/Data/ACW')

# # Extract specific file
# with rarfile.RarFile('archive.rar') as rf:
#     rf.extract('specific_file.txt', 'destination_folder')

# # List contents
# with rarfile.RarFile('archive.rar') as rf:
#     print(rf.namelist())