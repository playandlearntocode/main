from parsing.ll1 import LL1


def test():
    s = '{2;}'

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'block',

        'content': {
            'type': 'number',
            'value': '2'
        }
    }

    assert (ast == expected)
