'''
Solution for algorithmic problem: "Remove Invalid Parentheses", posted on LeetCode (problem 301).

Author:
Goran Trlin
Find this and more code examples on:
https://playandlearntocode.com

'''
class Solution:
    dict1 = {} # dictionary to help filtering the duplicate entries
    max_length = 0 # we're only interested in valid strings of maximum possible length

    def is_valid(self, unclosed):
        ''''
        Simple check of string validity, just based on the number of unclosed parentheses
        '''
        if unclosed == 0:
            return True
        else:
            return False

    def r1(self, s, s_current, unclosed_count, new_pos, new_char):
        '''
        Main recursive function for processing the input string.
        :param s: the original string
        :param s_current: current string, passed by the recursive caller
        :param unclosed_count: total number of unclosed parentheses
        :param new_pos: position of the next character in the original string
        :param new_char: next character in the original string
        :return:
        '''

        if self.is_valid(unclosed_count) == True:
            # print('Found valid:' + s_current)
            if (len(s_current) > self.max_length):
                self.max_length = len(s_current)
            # mark this key in the dict.:
            self.dict1[s_current] = 1

        # terminal condition:
        if new_pos > len(s) - 1:
            # print ('Exiting at pos.' + str(new_pos))
            return ''

        # boundary condition, end of string:
        next_char = ''
        if new_pos + 1 > len(s) - 1:
            next_char = ''
        else:
            next_char = s[new_pos + 1]

        if new_char == '(' or new_char == ')':
            # for "(" and ")", proceed with two paths - in one add the character, in the other one, omit it

            diff = 0 # whether to change the number of unclosed parentheses or not
            if new_char == '(':
                diff = 1

            if new_char == ')':
                diff = -1

            if unclosed_count < 0:
                # makes sure that invalid parentheses get filtered out:
                diff = 100

            self.r1(s, s_current, unclosed_count, new_pos + 1, next_char)
            self.r1(s, s_current + new_char, unclosed_count + diff, new_pos + 1, next_char)
        else:
            self.r1(s, s_current + new_char, unclosed_count, new_pos + 1, next_char)

    def removeInvalidParentheses(self, s: str):
        '''
        Main function
        Note: camelCase notation  in the name of this function is used just to keep in sync with the LeetCode assignment setup.
        :param s:
        :return:
        '''
        if len(s) == 0:
            return []

        # reset before the call:
        self.dict1 = {}
        self.max_length = 0

        # start the recursive call:
        self.r1(s, '', 0, 0, s[0])

        final_list = []
        # filter out too short valid pairs (only leave the ones with the min. number of parentheses removed):
        for key in self.dict1:
            if len(key) == self.max_length:
                final_list.append(key)

        return final_list
