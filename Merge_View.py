def merge():
	file1 = "user_view.txt"
	file2 = "../../20170112/extra_user_view.txt"
	file1 = open(file1).readlines()
	file2 = open(file2).readlines()
	file1.extend(file2)
	file1.sort(key=lambda x: int(x.split(',')[1]))
	with open("user_views.txt",'w') as fw:
		for l in file1:
			fw.write(l)
if __name__=='__main__':
	merge()
