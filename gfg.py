# Dynamic Programming | Set 16 (Floyd Warshall Algorithm)
def line_to_matrix(N,s):
    '''
    Args:
        s(list): 1D list of len N*N
        N(int): one dimension of the matrix
    Returns:
        list: 2D list of N*N dimension
    '''
    a = [[s[N*(i-1)+j-1] for j in range(1,N+1)] for i in range(1,N+1)]
    return a

t = int(input())
for _ in range(t):
    N = int(input())
    s = list(map(int, input().split()))
    dist = line_to_matrix(N,s)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    result = [dist[i][j] for i in range(N) for j in range(N)]
    print(' '.join(map(str, result)))

# Dynamic Programming | Set 17 Palindromic patitioning
t = int(input())
for _ in range(t):    
    a = input()
    N = len(a)
    f = [[0 for _ in range(N)] for _ in range(N)]
    is_palindrome = [[True for _ in range(N)] for _ in range(N)]    
    for length in range(1,N):
        for m in range(N-length):
            if a[m] == a[m+length] and is_palindrome[m+1][m+length-1]:
                is_palindrome[m][m+length] = True
            else:
                is_palindrome[m][m+length] = False
            
            if is_palindrome[m][m+length]:
                f[m][m+length] = 0
            else:
                f[m][m+length] = min(f[m][k] + f[k+1][m+length] + 1 for k in range(m,m+length))
    print(f[0][N-1])
