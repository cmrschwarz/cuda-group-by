import Data.Numbers.Primes

nextPrimesAfterPow2s :: Int -> [Int] -> [Int]
nextPrimesAfterPow2s pow2 (p:primes) = 
    if p > 2^pow2 
        then p : nextPrimesAfterPow2s (pow2 + 1) primes
        else nextPrimesAfterPow2s pow2 primes

main = mapM print (take 32 (nextPrimesAfterPow2s 0 primes))
