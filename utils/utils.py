import time

def write_list_to_file(filename, lst):
#     t=str(time.strftime("%c"))
#     filename = filename + '_' + t + '.txt'
#     print(filename)
    with open(filename, 'w') as f:
        for item in lst:
            f.write("%s\n" % item)