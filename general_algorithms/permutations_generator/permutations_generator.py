class PermutationsGenerator:
    all_permutations = []

    def _backtrack(self, cur_arr, search_space):
        '''
        Loop over the remaining search space, and add every element to the beginning of a new branch:
        '''
        if len(search_space) == 0:
            # the cur_arr is now full so we can add this combination to the output queue:
            self.all_permutations.append(cur_arr)
            return

        for i in range(len(search_space)):
            # contactenate the rest of the elements into the new search_space
            self._backtrack(cur_arr[:] + [search_space[i]], search_space[:i] + search_space[i + 1:])
        return

    def permute(self, nums):
        self.all_permutations = []

        # start by adding N branches as the starting points:
        for i in range(len(nums)):
            self._backtrack([nums[i]], nums[:i] + nums[i + 1:])

        return self.all_permutations
