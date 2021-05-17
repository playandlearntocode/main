from permutations_generator import PermutationsGenerator

# MAIN TESTING PROGRAM:
test_input = [1,2,3]

pg = PermutationsGenerator()
result = pg.permute(test_input)

print('Input array:')
print(test_input)
print('Permutations:')
print(result)