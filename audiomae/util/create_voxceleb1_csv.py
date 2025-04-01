def create_csv():
    # Specify the output file name
    output_filename = "voxceleb1_label.csv"
    
    # Open the file in write mode
    with open(output_filename, "w", encoding="utf-8") as f:
        # Go from 0 to 1251 (inclusive)
        f.write("index,mid,display_name\n")
        for i in range(1252):
            # Generate zero-padded index starting from 1 for the vc and sid parts
            vc_id = f"{i+1:04d}"     # e.g., "0001", "0002", ...
            sid_id = f"{i+1:04d}"    # same zero-padded ID for sid
            
            # Build the line: index, /m/vcXXXX, "sid-XXXX"
            line = f"{i},/m/vc{vc_id},\"sid-{sid_id}\"\n"
            
            # Write the line to the file
            f.write(line)

if __name__ == "__main__":
    create_csv()
    print("CSV file created successfully!")
