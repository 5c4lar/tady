Generated using https://vmpsoft.com/uploads/VMProtectDemo.tar.gz
With the Examples/Licensing/BCB/TestApp.exe as an example
In addition to open-source obfuscators, we also evaluate the disassemblers against one famous commercial 
obfuscator VMProtect.
This obfuscator support mutation, virtualization or Ultra(Combine the previous two methods).
Where mutation obfuscate the binary using the same ISA instructions, virtualization introduce 
a virtual machine and encode the protected function into byte sequences of that virtual machine,
such strong obfuscation is beyond our scope. We only test the disassemblers against mutation.
To ease evaluation, we also disabled the "Pack the output file"
option, which will compress the output binary and only decompress at runtime, which is an anti measure to 
static analysis, which is also beyond our scope. In this work, we focus on analyze the statically obfuscated
code. Under such configuration, we obfuscate one of the example binary provided by VMProtect Demo,
Examples/Licensing/BCB/TestApp.exe. Labeling the ground truth of obfuscated binary is much labor, and hard
to be done at large scale, since the obfuscation pattern is common, we obfuscate only the first one of the 
protected functions btTryClick and analyze it manually to collect ground truth for this function.