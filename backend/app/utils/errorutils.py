import time

def repeat_until_finish(func, max_retries=5, delay=0.1):
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            time.sleep(delay)
    return None