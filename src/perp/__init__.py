"""The perpetual-futures bot.

Its own package because RL-019 makes each segment its own bot with its own
architecture, data and features. Shared machinery - the store, the paper broker,
the feature modules - stays where it is; what lives here is what is true about
perpetuals and nothing else.
"""
