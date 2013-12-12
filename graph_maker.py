#!/usr/bin/env python
# a stacked bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt


# N = 5
# # Serial
# to_YIQ = [8.382845, 6.619563, 8.381515, 3.07796, 9.9618]
# feature = [17.602839, 13.158485, 17.438558, 6.112318, 20.771632]
# ann = [2.637077, 2.545599, 2.62029, 1.650415, 4.902459]
# coherence = [50.579986, 64.041284, 51.420666, 51.476018, 138.471253]
# to_RGB = [6.318207, 7.714812, 6.35584, 5.949681, 16.160461]

# N = 6
# # Parallel
# to_YIQ = [2.119645,2.115798,2.108119,2.137708,2.098378,2.172944]
# feature = [2.149597,2.140362,2.133806,2.11893,2.140967,2.477246]
# ann = [2.279708,1.808282,2.239211,1.33596,3.980359,27.489703]
# coherence = [16.6554560661,17.1107981205,17.2222790718,15.5332331657,43.6365399361,305.911691904]
# to_RGB = [0.495962,0.495347,0.493482,0.465032,1.235267,8.599976]

N = 6
# Parallel
to_YIQ = [2.121109,2.119645,2.115798,2.098378,2.16056,2.172944]
feature = [2.127265,2.149597,2.140362,2.140967,2.256579,2.477246]
ann = [2.127265,2.149597,2.140362,2.140967,2.256579,2.477246]
coherence = [14.9180510044,16.6554560661,17.1107981205,43.6365399361,122.29063201,305.911691904]
to_RGB = [0.436488,0.495962,0.495347,1.235267,3.446585,8.599976]

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, to_RGB,   width, color='r')
p2 = plt.bar(ind, coherence, width, color='y',
             bottom=to_RGB)
p3 = plt.bar(ind, ann, width, color='g', bottom=[sum(x) for x in zip(to_RGB, coherence)])
p4 = plt.bar(ind, feature, width, color='magenta', bottom=[sum(x) for x in zip(to_RGB, coherence, ann)])
p5 = plt.bar(ind, to_YIQ, width, color='cyan', bottom=[sum(x) for x in zip(to_RGB, coherence, ann, feature)])


plt.ylabel('Time (seconds)')
plt.title('Parallel Running Time for Blur Analogies of Different Sizes (4 Cores)')
# plt.xticks(ind+width/2., ('Blur(flower)', 'Art(rose)', 'Emboss(flower)', 'Superres', 'Art(Miley)', 'Art(Tiger)') )
plt.xticks(ind+width/2., ('253x384', '362x300', '373x350', '430x650', '768x1024', '1200x1600') )
plt.yticks(np.arange(0,350,10))
plt.legend( (p5[0], p4[0], p3[0], p2[0], p1[0]), ('To YIQ', 'Feature', 'ANN', 'Coherence', 'To RGB'), loc='upper left' )

plt.show()