from numpyVision import NumpyVision
def main():
    n1 = NumpyVision("images.png")
    n1.blur(10,inplace = False).show()
    

if __name__ == "__main__":
    main()