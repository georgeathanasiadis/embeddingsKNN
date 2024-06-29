import json

# File paths
input_file_path = "OLD_LFR_N10000_ad5_mc50_mu0.7_gt.txt"
output_file_path = "LFR_N10000_ad5_mc50_mu0.7_gt.txt"

with open(input_file_path, "r") as file:
    data = json.loads(file.read())  #ensure the input is properly formatted as a JSON list of lists

def format_list(data):
    formatted_output = ""
    for index, sublist in enumerate(data, start=1):
        formatted_output += f"{index}: {sublist}\n"
    return formatted_output

#get formatted output
formatted_output = format_list(data)

#print the formatted output (optional)
print(formatted_output)

#save the formatted output to the output file
with open(output_file_path, "w") as file:
    file.write(formatted_output)

print(f"Formatted data saved to {output_file_path}")
