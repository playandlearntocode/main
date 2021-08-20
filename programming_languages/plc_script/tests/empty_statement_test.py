from parsing.ll1 import LL1


def test():
    s = ';'

    ll1 = LL1(s)
    ast = ll1.parse()

    expected = {
        'type': 'empty_statement'
    }

    assert (ast == expected)
