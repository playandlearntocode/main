class Solution:
    MIN_INT = pow(-2, 31)
    MAX_INT = pow(2, 31) - 1

    def recursive_divide(self, current_dividend: int, current_divisor: int):
        quotient = 1
        accumulator = current_divisor  # same as current_divisor * quotient because quotient == 1 at this point
        # base case
        if current_dividend < current_divisor:
            return 0
        elif current_dividend == current_divisor:
            return 1

        while accumulator < current_dividend:
            quotient = quotient << 1
            accumulator = accumulator << 1  # implicit quotient inclusion here!

        # undo the last step, because accumulator is now larger than current_dividend
        accumulator = accumulator >> 1
        quotient = quotient >> 1
        return quotient + self.recursive_divide(current_dividend - accumulator, current_divisor)

    def divide(self, dividend: int, divisor: int) -> int:
        '''
        Main method of this module.
        :param dividend:
        :param divisor:
        :return:
        '''
        # determine the sign of quotient:
        negative = False
        if (dividend >= 0 and divisor >= 0):
            negative = False
        elif (dividend < 0 and divisor >= 0):
            negative = True
        elif (dividend > 0 and divisor <= 0):
            negative = True

        # extract positive values of dividend and divisor:
        abs_dividend, abs_divisor = abs(dividend), abs(divisor)

        # watch for limits:
        if (abs_divisor == 1):
            if (negative == True):
                return -abs_dividend if Solution.MIN_INT < -abs_dividend else Solution.MIN_INT
            else:
                return abs_dividend if Solution.MAX_INT > abs_dividend else Solution.MAX_INT

        q = self.recursive_divide(abs_dividend, abs_divisor)
        return q if negative == False else -q
