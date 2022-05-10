def printProgress(stop):
    states = [' | ', ' / ', ' — ', ' \\ ']
    nextc = 0
    while True:
        sys.stdout.write('\r[{}]'.format(states[nextc]))
        sys.stdout.flush()
        time.sleep(2.5)
        nextc = nextc + 1 if nextc < 3 else 0
        if stop():
            sys.stdout.write('\r[{}]'.format('OK!'))
            sys.stdout.flush()
            break

    sys.stdout.write('\n')
    return

