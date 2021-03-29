from .solution import Solution

# MAIN TESTING PROGRAM:
test_input_string = '()())()'
sln = Solution()
result = sln.removeInvalidParentheses(test_input_string)

print('Input string:')
print(test_input_string)
print('RESULTS:')
print(result)