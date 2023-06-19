# Plot the training and validation losses.
import matplotlib.pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA500', '#800080', '#008080']

colors_dark = ['#00FFFF', '#FF1493', '#00FF00', '#FF4500', '#ADFF2F', '#FF00FF', '#1E90FF', '#FF69B4', '#20B2AA', '#FF8C00']


def plotMultiple( X,  xlabel, ylabel,title, name, styleDark = False ):
    
    plt.figure()
    if(styleDark):
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axarr = plt.subplots(figsize=(20, 10), dpi= 80)
    plt.title(title,size=40)
    plt.xlabel(xlabel,size=30)
    plt.ylabel(ylabel,size=30)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('font', family='serif')

    #create a funtion that iterates over the list of lists and plots each one
    for i,row in enumerate(X):
        x = row.x
        y = row.y
        try:
            color = row.color
        except:
            if(styleDark):
                color = colors_dark[i]
            else:
                color = color[i]


    

        plt.plot(x, color=color, linewidth =3, label=f'{row.label}' )
        plt.plot(y, color=color, linewidth =3, label=f'{row.label}' )
    
    plt.legend(fontsize="20", loc ="upper left")
    plt.savefig(f'./plots/{name}.png')
    
    plt.show()

def plotMatrix(M,xlabel, ylabel,title, name, styleDark = False):

    plt.figure()
    if(styleDark):
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axarr = plt.subplots(figsize=(20, 10), dpi= 80)
    plt.title(title,size=40)
    plt.xlabel(xlabel,size=30)
    plt.ylabel(ylabel,size=30)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('font', family='serif')

    cmap = plt.cm.get_cmap('Greys_r', 256)

    plt.imshow(M, cmap=cmap, interpolation='nearest')
    plt.colorbar()

    
    plt.savefig(f'./plots/{name}.png')
    
    plt.show()
    
