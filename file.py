# fp = open("user_pay.txt",'r')
# with open("user_pay_100.txt",'a+') as fw:
# 	#c = 0
# 	for i in xrange(100):
# 		fw.write( (fp.readline()))
# this is a comment

# fp.close()
fp = open("user_pay.txt",'r')
a = [0,0,0,0,0]
for line in fp.readlines():
	#print 'start'
	terms = line.strip('\n\r').split(',')
	for i in range(0,2):
		try:
			t = int(terms[i])
			if t>a[i]:
				a[i] = t
		except:
			pass
print a
