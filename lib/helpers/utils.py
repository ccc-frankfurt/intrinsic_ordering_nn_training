import subprocess


class GPUMem:
    def __init__(self, is_gpu, device_id):
        """
        reads the total gpu memory that the program can access and computes the amount of gpu memory available

        Parameters:
            is_gpu (bool): says if the computation device is cpu or gpu

        Attributes:
            total_mem (float): total gpu memory that the program can access
        """
        self.is_gpu = is_gpu
        self.device_id = device_id
        if self.is_gpu:
            self.total_mem = self._get_total_gpu_memory()

    def _get_total_gpu_memory(self):
        """
        gets the total gpu memory that the program can access

        Returns:
            total gpu memory (float) that the program can access
        """
        total_mem = subprocess.check_output(["nvidia-smi", "--id="+str(self.device_id), "--query-gpu=memory.total",
                                             "--format=csv,noheader,nounits"])

        return float(total_mem[0:-1])  # gets rid of "\n" and converts string to float

    def get_mem_util(self):
        """
        gets the amount of gpu memory currently being used

        Returns:
            mem_util (float): amount of gpu memory currently being used
        """
        if self.is_gpu:
            # Check for memory of GPU ID 0 as this usually is the one with the heaviest use
            free_mem = subprocess.check_output(["nvidia-smi", "--id="+str(self.device_id), "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"])
            free_mem = float(free_mem[0:-1])    # gets rid of "\n" and converts string to float
            mem_util = 1 - (free_mem / self.total_mem)
        else:
            mem_util = 0
        return mem_util