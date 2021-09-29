from multiprocessing import Pool,Process,Queue
from functools import wraps
import os,time,random

global_counter = 0
def log_pid(func):
    '''
    Decorator that reports the pid.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        global global_counter
        global_counter += 1
        pid = os.getpid()
        print(f"this is the {global_counter} time,pid is {pid}")
        result = func(*args, **kwargs)

        return result
    return wrapper

@log_pid
def task(msg,arg2,arg3=1):
    """测试多进程"""
    start_time = time.time()
    print(f'start execute process{os.getpid()},it is the {msg} loop')
    time.sleep(random.random()*4)
    stop_time = time.time()
    print(f'execution finished, time spend is {stop_time-start_time}')
    return msg+arg2+arg3

def main():
    pool = Pool(processes=5)
    K=10
    results= []
    for i in range(1,K+1):
        # res = minimize(fun=regularized_cost, x0=theta0, args=(X, y_mask), method='Newton-CG', jac=regularized_gradients)
        print(f"add task{i}")
        result = pool.apply_async(task,kwds={'msg':i,"arg2":0,"arg3":2})
        results.append(result)
    pool.close() 
    pool.join()
    print(results)
    print(results[0].get())

def prime_factor(value):
    factors = []
    for divisor in range(2, value-1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
        else:
            factors = [value]
    return factors


def prime_factor2(value,arg2,arg3=1):
    factors = []
    for divisor in range(2, value-1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
        else:
            factors = [value]
    return factors+arg2+arg3


from scipy.optimize import minimize

if __name__ == "__main__":
    main()