'''
AST interpreter for plc_script
Author:
Goran Trlin
https://playandlearntocode.com
'''


class ASTInterpreter:
    # START OF BUILTIN FUNCTIONS
    def printIt(s):
        # note that no self parameter :)
        print(s)
    # END OF BUILTIN FUNCTIONS

    # root environment:
    global_env = {'_parent': None, 'printIt': printIt}

    # main entry point to the interpreter:
    def program(self, ast):
        return self.statement_list(ast, self.global_env)

    def statement_list(self, arr, env):
        last_result = None
        if isinstance(arr, list):
            for i in range(len(arr)):
                last_result = self.statement(arr[i], env)
            return last_result
        else:
            return self.statement(arr, env)

    def statement(self, root, env):
        if root['type'] == 'function_declaration_statement':
            return self.function_declaration_statement(root, env)
        elif root['type'] == 'if_statement':
            return self.if_statement(root, env)
        elif root['type'] == 'while_statement':
            return self.while_statement(root, env)
        elif root['type'] == 'block':
            return self.block_statement(root, env)
        elif root['type'] == 'return_statement':
            return self.return_statement(root, env)
        else:
            return self.expression_statement(root, env)

    def assignment_expression(self, root, env):
        left = root['content']['left']
        right = root['content']['right']
        rightVal = self.expression(right, env)

        if (left['type'] != 'identifier'):
            raise Exception('Identifier expected on the left side!')

        env[left['value']] = rightVal

        return rightVal

    def multiplication_expression(self, root, env):
        left = root['content']['left']
        right = root['content']['right']

        leftVal = self.expression(left, env)
        rightVal = self.expression(right, env)

        return leftVal * rightVal

    def division_expression(self, root, env):
        left = root['content']['left']
        right = root['content']['right']

        leftVal = self.expression(left, env)
        rightVal = self.expression(right, env)

        return leftVal / rightVal

    def addition_expression(self, root, env):
        left = root['content']['left']
        right = root['content']['right']
        leftVal = self.expression(left, env)
        rightVal = self.expression(right, env)

        if type(leftVal) == int and type(rightVal) == int:
            # numbers:
            return leftVal + rightVal
        else:
            # strings
            return leftVal + rightVal

    def subtraction_expression(self, root, env):
        left = root['content']['left']
        right = root['content']['right']
        leftVal = self.expression(left, env)
        rightVal = self.expression(right, env)

        if type(leftVal) == int and type(rightVal) == int:
            # numbers:
            return leftVal - rightVal
        else:
            # strings
            return leftVal - rightVal

    def block_statement(self, root, env, createEnv=True):
        if createEnv == True:
            newEnv = {'_parent': env}
        else:
            newEnv = env

        last_result = None
        if isinstance(root['content'], list):
            for i in range(len(root['content'])):
                last_result = self.statement(root['content'][i], newEnv)
        else:
            last_result = self.statement(root['content'], newEnv)
        return last_result

    def return_statement(self, root, env):
        res = self.expression(root['content'], env)
        return res

    def expression(self, root, env):
        if root['type'] == 'function_call_expression':
            return self.function_call(root, env)
        elif root['type'] == 'operator_equals':
            return self.operator_equals(root, env)
        elif root['type'] == 'operator_less_than':
            return self.operator_less_than(root, env)
        elif root['type'] == 'operator_greater_than':
            return self.operator_greater_than(root, env)
        elif root['type'] == 'assignment_expression':
            return self.assignment_expression(root, env)
        elif root['type'] == 'multiplication_expression':
            return self.multiplication_expression(root, env)
        elif root['type'] == 'division_expression':
            return self.division_expression(root, env)
        elif root['type'] == 'addition_expression':
            return self.addition_expression(root, env)
        elif root['type'] == 'subtraction_expression':
            return self.subtraction_expression(root, env)
        elif root['type'] == 'identifier':
            return self.identifier(root, env)
        else:
            return self.literal(root, env)

    def operator_equals(self, root, env):
        left = self.expression(root['content']['left'], env)
        right = self.expression(root['content']['right'], env)
        return left == right

    def operator_less_than(self, root, env):
        left = self.expression(root['content']['left'], env)
        right = self.expression(root['content']['right'], env)
        return left < right

    def operator_greater_than(self, root, env):
        left = self.expression(root['content']['left'], env)
        right = self.expression(root['content']['right'], env)
        return left > right

    def expression_statement(self, root, env):
        return self.expression(root, env)

    def function_declaration_statement(self, root, env):
        env[root['name']] = root
        return env[root['name']]

    def if_statement(self, root, env):
        # newEnv = {'_parent': env}
        newEnv = env
        condition = self.expression(root['condition'], env)
        if (condition == True):
            newEnv = {'_parent': env}
            return self.block_statement(root['case_true'], newEnv, False)
        else:
            newEnv = {'_parent': env}
            if (root['case_false'] is not None):
                return self.block_statement(root['case_false'], newEnv, False)

    def while_statement(self, root, env):

        last_run = None
        # newEnv = {'_parent': env}
        newEnv = env
        condition = self.expression(root['condition'], env)

        while (condition == True):
            last_run = self.block_statement(root['while_body'], newEnv, False)
            condition = self.expression(root['condition'], newEnv)  # recheck
        return last_run

    def resolve_name(self, env, name):
        if (name not in env):
            if (env.get('_parent') is not None):
                return self.resolve_name(env['_parent'], name)
            else:
                return None
        else:
            return env[name]

    def function_call(self, root, env):
        callee = self.resolve_name(env, root['callee'])
        if callee is None:
            raise Exception('Function not defined!')

        newEnv = {'_parent': env}

        funDefinitionObj = callee

        if (callable(funDefinitionObj)):
            # builtin python function:
            fArgs = []
            for i in range(len(root['arguments'])):
                curArgValue = self.expression(root['arguments'][i], env)
                fArgs.append(curArgValue)

            # unpack argument list and call:
            return (funDefinitionObj(*fArgs))
        else:
            if (funDefinitionObj['arguments'] is not None and len(funDefinitionObj['arguments']) > 0):
                for i in range(len(funDefinitionObj['arguments'])):
                    curArgName = funDefinitionObj['arguments'][i]['value']
                    curArgValue = self.expression(root['arguments'][i], env)
                    newEnv[curArgName] = curArgValue

            return self.block_statement(funDefinitionObj['content'], newEnv, False)

    def identifier(self, root, env):
        ret = self.resolve_name(env, root['value'])
        return ret

    def literal(self, root, env):
        if root['type'] == 'number':
            return self.number(root, env)
        elif root['type'] == 'string':
            return self.string(root, env)

        else:
            raise Exception('Invalid literal ' + root['type'])

    def number(self, root, env):
        return int(root['value'])

    def string(self, root, env):
        return root['value']

    def interpret(self, ast):
        return self.program(ast)
