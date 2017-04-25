import module_test
import module_test2.echo as echo
from module_test2.echo import go2

print ("sum:",module_test.go(3,2))
go2()
echo.go3()