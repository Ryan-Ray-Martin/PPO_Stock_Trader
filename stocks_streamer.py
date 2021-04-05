import os
import json
import datetime
import asyncio
import uvloop
import logging
from polygon_streamer import PolygonStreamer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class StocksStreamer(PolygonStreamer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	"""{
 "ev": "A",
 "sym": "SPCE",
 "v": 200,
 "av": 8642007,
 "op": 25.66,
 "vw": 25.3981,
 "o": 25.39,
 "c": 25.39,
 "h": 25.39,
 "l": 25.39,
 "a": 25.3714,
 "z": 50,
 "s": 1610144868000,
 "e": 1610144869000
}"""

	async def callback(self, message_str):
		print(message_str)


def main():
	polygon_api_key = "x6XqEaDjOWJXwVQRyBQ5kMEi8KlCYZqo"
	cluster = "/stocks"
	symbols_str = "A.AAPL, A.NIO, A.RIOT"
	streamer = StocksStreamer(api_key=polygon_api_key, cluster=cluster, symbols_str=symbols_str)
	streamer.start()


if __name__ == '__main__':
	main()
