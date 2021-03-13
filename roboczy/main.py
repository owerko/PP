
my_dict = {'x':500, 'y':5874, 'z': 560}

key_max = max(my_dict.keys(), key=(lambda k: my_dict[k]))
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))

print(my_dict.keys())
print(max(my_dict.values()))

print('Maximum Value: ',my_dict[key_max])
print('Minimum Value: ',my_dict[key_min])