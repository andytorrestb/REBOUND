import time
import os
import re
import matplotlib.pyplot as plt



def parse_simulation_data(lines):
    """Extracts relevant simulation data from a block of lines."""
    data = {
        "Time": None,
        "Number of dsmc particles": None,
        "Average linear kinetic energy": None,
        "Average internal energy": None,
        "Average total energy": None,
        "Mass in system": None,
        "ClockTime": None
    }

    for line in lines:
        # print(line)
        # input("Press Enter to continue...")
        if line.strip().startswith("Time ="):
            data["Time"] = line.strip().split("=", 1)[1].strip()
        elif "Number of dsmc particles" in line:
            dsmc_match = re.search(r'Number of dsmc particles\s*=\s*([0-9.eE+-]+)', line)            
            # print(dsmc_match)
            if dsmc_match:
                data["Number of dsmc particles"] = dsmc_match.group(1)

        elif "Average linear kinetic energy" in line:
            # Example: Average linear kinetic energy = 0.000123 J
            lin_kin_match = re.search(r'Average linear kinetic energy\s*=\s*([0-9.eE+-]+)(?:\s*J)?', line)
            if lin_kin_match:
                data["Average linear kinetic energy"] = lin_kin_match.group(1)

        elif "Average internal energy" in line:
            # Example: Average internal energy = 0.000456 J
            int_energy_match = re.search(r'Average internal energy\s*=\s*([0-9.eE+-]+)(?:\s*J)?', line)
            if int_energy_match:
                data["Average internal energy"] = int_energy_match.group(1)
        
        elif "Average total energy" in line:
            # Example: Average total energy = 0.000789 J
            total_energy_match = re.search(r'Average total energy\s*=\s*([0-9.eE+-]+)(?:\s*J)?', line)
            if total_energy_match:
                data["Average total energy"] = total_energy_match.group(1)

        elif "Mass in system" in line:
            # Example: total mass = 0.001 kg
            mass_match = re.search(r'Mass in system\s*=\s*([0-9.eE+-]+)', line)
            if mass_match:
                data["Mass in system"] = mass_match.group(1)

        elif "ExecutionTime" in line and "ClockTime" in line:
            # Example: ExecutionTime = 0.52 s  ClockTime = 1 s
            exec_match = re.search(r'ExecutionTime\s*=\s*([0-9.eE+-]+)\s*s', line)
            clock_match = re.search(r'ClockTime\s*=\s*([0-9.eE+-]+)\s*s', line)
            if exec_match:
                data["ExecutionTime"] = exec_match.group(1)
            if clock_match:
                data["ClockTime"] = clock_match.group(1)
    return data

def tail_latest_timestep(filepath, num_lines=200):
    try:
        with open(filepath, 'rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            blocksize = 1024
            data = b''
            while filesize > 0 and data.count(b'\n') < num_lines:
                read_size = min(blocksize, filesize)
                filesize -= read_size
                f.seek(filesize)
                data = f.read(read_size) + data
            lines = data.decode(errors='ignore').splitlines()
        
        # Find the latest "Time =" line
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip().startswith("Time ="):
                latest_index = i
                break
        else:
            print("No time steps found in the log file.")
            return
        # Extract the block for the latest time step
        block = lines[latest_index:]
        sim_data = parse_simulation_data(block)
        print(f"--- Latest time step ({time.strftime('%H:%M:%S')}) ---")
        print(block)
        for key, value in sim_data.items():
            # if value is not None:
            print(f"{key}: {value}")

        return sim_data
    except FileNotFoundError:
        print(f"File not found: {filepath}")

def monitor_file(filepath, interval=10):

    data_arrays = {
        "Time": [],
        "Number of dsmc particles": [],
        "Average linear kinetic energy": [],
        "Average internal energy": [],
        "Average total energy": [],
        "Mass in system": [],
        "ClockTime": []
    }

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'bo-')
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of dsmc particles")
    ax.set_title("Time vs Number of dsmc particles")

    while True:
        sim_data = tail_latest_timestep(filepath)

        # Only proceed if all values in sim_data are not None
        if sim_data and all(value is not None for value in sim_data.values()):
            for key in data_arrays:
                value = sim_data[key]
                # Convert to float if possible, else keep as is (e.g., None)
                # try:
                value = float(value)
                # except (TypeError, ValueError):
                #     value = 0
                data_arrays[key].append(value)
        
        # Update plot
        line.set_xdata(data_arrays["Time"])
        line.set_ydata(data_arrays["Number of dsmc particles"])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        print("\n--- Data Arrays ---")
        for key, values in data_arrays.items():
            print(f"{key}: {values[-5:]}")

        time.sleep(interval)

if __name__ == "__main__":
    filepath = "log.dsmcFoam"
    monitor_file(filepath, 10)