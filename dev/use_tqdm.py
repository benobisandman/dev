from tqdm import tqdm
import time
from random import randint

# Example list of tickers (you can replace this with your actual data)
ticker_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Simulating some work with random sleep time
def process_ticker(ticker):
    # Generate a random sleep time between 1 and 5 seconds
    sleep_time = randint(1, 5)
    time.sleep(sleep_time)
    print(f"Processed {ticker} (Slept for {sleep_time} seconds)")

# Iterate through the ticker_list with tqdm
for ticker in tqdm(ticker_list, desc="Processing tickers"):
    process_ticker(ticker)