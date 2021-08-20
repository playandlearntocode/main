from parsing.ll1 import LL1


def test():
    s = '''
    a = 2;
    
    if (a == 2) {
        printIt("a is 2");
    } else {
      printIt("a is not 2");
    }
    
    '''

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = [
        {
            'type': 'assignment_expression',
            'content': {
                'left': {
                    'type': 'identifier',
                    'value': 'a'
                },
                'right': {
                    'type': 'number',
                    'value': '2'
                }
            }
        },
        {
        'type': 'if_statement',
        'condition': {
            'type': 'operator_equals',
            'content': {
                'left': {
                    'type': 'identifier',
                    'value': 'a'
                },
                'right': {
                    'type': 'number',
                    'value': '2'
                }
            }
        },
        'case_true': {
            'type': 'block',
            'content': {
                'type': 'function_call_expression',
                'callee': 'printIt',
                'arguments': [
                    {'type': 'string',
                     'value': 'a is 2'
                     }
                ]
            }
        }
        ,
        'case_false': {
            'type': 'block',
            'content': {
                'type': 'function_call_expression',
                'callee': 'printIt',
                'arguments': [
                    {'type': 'string',
                     'value': 'a is not 2'
                     }
                ]
            }
        }

    }
    ]

    assert (ast == expected)
