import streamlit as st
import matplotlib.pyplot as plt

def DDALine (x1, y1, x2, y2, color):
    dx = x2 - x1
    dy = y2 - y1
    steps = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    Xinc = float(dx / steps)
    Yinc = float(dy / steps)
    mid_x = x1 + (dx / 2)
    mid_y = y1 + (dy / 2)

    #Create a new figure to plot the DDA line
    fig = plt.figure()
    for i in range(0, int(steps + 1)):
        plt.plot(int(x1), int(y1), color)
        x1 += Xinc
        y1 += Yinc
    
    #Plotting the midpoint for the DDALine
    plt.plot(int(mid_x), int(mid_y), "g.")
    plt.title("DDA Line with Mid Point")
    st.pyplot(fig)

def BresenhamLine(x1, y1, x2, y2, color):
    dx = x2 - x1
    dy = y2 - y1
    steps = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    Xinc = int(dx / steps)
    Yinc = int(dy / steps)
    mid_x = x1 + (dx / 2)
    mid_y = y1 + (dy / 2)

    #Create a new figure to plot the Bresenham line
    fig = plt.figure()
    x = x1
    y = y1
    for i in range(0, int(steps + 1)):
        plt.plot(int(x), int(y), color)
        x += Xinc
        y += Yinc
    
    #Plotting the midpoint for the bresenham
    plt.plot(int(mid_x), int(mid_y), "g.")
    plt.title("Bresenham Line with Mid Point")
    st.pyplot(fig)

def main():
    x = st.number_input("Enter X1: ")
    y = st.number_input("Enter Y1: ")
    xEnd = st.number_input("Enter X2: ")
    yEnd = st.number_input("Enter Y2: ")
    color = "r."

    DDALine(x, y, xEnd, yEnd, color)
    BresenhamLine(x, y, xEnd, yEnd, color)

if __name__ == '__main__':
    main()
