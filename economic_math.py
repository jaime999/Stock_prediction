import talib

def getRSI(ultimaCotizacion):
    print(ultimaCotizacion)
    rsi = talib.RSI(ultimaCotizacion)
    return rsi
    print(rsi)
