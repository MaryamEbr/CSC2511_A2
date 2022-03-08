def count_last_character_repeats(s: str, ch: str) -> int:
    """Return the number of times the character ch appears at the END of the
    string s.

    Precondition: len(ch) == 1

    >>> count_last_character_repeats("buzz", "z")
    2
    >>> count_last_character_repeats("buzzz", "z")
    3
    >>> # TODO: (1) Add example that demonstrates that current function body
    >>> #           is incorrect.
    >>>count_last_character_repeats("buzaz", "z")
    1
    """

    # TODO: (2) Modify the function body to make it correct.

    count_ch = 0
    i = len(s) - 1
    while i >= 0:
        if ch == s[i]:
            count_ch = count_ch + 1
            i = i - 1
        else:
            return count_ch

    return count_ch
print(count_last_character_repeats("buzjinglezzz", "z"), " 3")
print(count_last_character_repeats("buzz", "z"), " 2")
print(count_last_character_repeats("buzzz", "z"), " 3")
print(count_last_character_repeats("Digggngggg", "g") , " 4")
print(count_last_character_repeats("buzzzrr", "z"), " 0")