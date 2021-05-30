class Solution:
    '''
    Dynamic programming solution O(n)
    After trivial cases i=0 and i=1 are resolved:
    At each step, a decision is made, either to carry over f(-1) or to sum f(-1) and f(-2) or even to
    stop altogether

    Write the input sequence 1111 on paper and go step by step, to see the options at each step
    '''

    def dynamic_compute(self, s):
        table = [0] * len(s)

        if s[0] != '0':
            table[0] = 1
        else:
            table[0] = 0

        if len(s) == 1:
            return table[0]

        double = s[0] + s[1]
        double_num = int(double)

        if double[0] == '0' or double_num > 26:
            double = None

        if s[0] == '0':
            table[0] = 0
            table[1] = 0
        elif s[1] == '0':
            if double != None:
                table[0] = 1
                table[1] = 1
        else:
            if double != None:
                table[0] = 1
                table[1] = 2
            else:
                table[0] = 1
                table[1] = 1

        for i in range(2, len(s)):
            single = s[i]

            if i < 1:
                double = '999'
                double_num = 999
            else:
                double = s[i - 1] + s[i]
                double_num = int(double)

            if double == '00':
                return 0

            if double[0] == '0' or double_num > 26:
                double = None

            if single == '0':
                single = None

            if single == None and double != None:
                table[i] = table[i - 2]
            elif single == None and double == None:
                table[i] = 0
            elif double != None:
                table[i] = table[i - 1] + table[i - 2]
            elif single != None:
                table[i] = table[i - 1]
            else:
                table[i] = table[i - 1]

        return table[len(s) - 1]

    # LeetCode camelCase:
    # def numDecodings(self, s: str) -> int:
    def num_decodings(self, s: str) -> int:
        return self.dynamic_compute(s)
