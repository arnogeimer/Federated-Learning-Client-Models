start cmd /k C:\Users\arno.geimer\AppData\Local\Programs\Python\Python310\python.exe C:\Users\arno.geimer\Desktop\Paper\code\GLM_server.py 
for /l %%x in (1, 1, 5) do (
start cmd /k C:\Users\arno.geimer\AppData\Local\Programs\Python\Python310\python.exe C:\Users\arno.geimer\Desktop\Paper\code\GLM_client.py 
)
%* 
pause