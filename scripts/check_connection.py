"""Read-only connection check before a live extraction run.

Answers one question in plain words: can this laptop reach the AI providers'
servers right now? It looks up the server names and opens a bare connection.
No request is sent, no API key is read, nothing is written. Safe to run as
often as needed; it changes nothing.
"""
import socket
import sys

HOSTS = [('Gemini', 'generativelanguage.googleapis.com'),
         ('OpenAI', 'api.openai.com')]


def resolve(name):
    # The DNS "phone book" step: turn the server name into addresses.
    try:
        found = socket.getaddrinfo(name, 443)
    except OSError as error:
        return None, error
    return sorted({item[4][0] for item in found}), None


def connect(name, timeout=10):
    # Open a bare TCP connection on the HTTPS port, then close it at once.
    try:
        with socket.create_connection((name, 443), timeout=timeout):
            return None
    except OSError as error:
        return error


def main():
    problems = 0
    for label, name in HOSTS:
        addresses, error = resolve(name)
        if error is not None:
            problems += 1
            print('PROBLEM: cannot look up ' + name + ' (' + type(error).__name__
                  + ' errno ' + str(getattr(error, 'errno', None)) + ')')
            print('  The DNS lookup is failing, so the laptop cannot find the '
                  + label + ' server. Nothing was sent.')
            print('  Try a phone hotspot, or check the DNS settings, then run this again.')
            continue
        print('OK: ' + name + ' resolves (' + str(len(addresses)) + ' addresses)')
        error = connect(name)
        if error is not None:
            problems += 1
            print('PROBLEM: ' + name + ' resolves but the connection will not open ('
                  + type(error).__name__ + ' errno ' + str(getattr(error, 'errno', None)) + ')')
            print('  A firewall or the network is blocking it. Nothing was sent.')
        else:
            print('OK: a connection to the ' + label + ' server opens (no request sent, no key used)')
    print()
    if problems:
        print('Connection is NOT ready. Fix the problem above, then run this check again.')
        return 1
    print('Connection looks ready. You can run the extraction command when you choose.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
