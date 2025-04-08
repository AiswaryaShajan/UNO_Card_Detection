def main():
    print("UNO Card Detection")
    print("1. Use Image")
    print("2. Use Webcam")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        from card_detection import detect_card_from_image
        detect_card_from_image()
    elif choice == '2':
        from card_detection import detect_card_from_webcam
        detect_card_from_webcam()
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
