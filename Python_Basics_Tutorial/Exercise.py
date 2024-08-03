#求1-100中所有素数的和
# define a function to judge whether a number is a prime
def isPrime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    s = 0
    for i in range(2, 101):
        if isPrime(i):
            s += i
    print(s)