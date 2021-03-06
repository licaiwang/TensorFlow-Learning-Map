

## 機器學習介紹

![](res/chapter1-1.png)

這門課，我們預期可以學到什麼呢？我想多數同學的心理預期就是你可以學到一個很潮的人工智慧。我們知道，從今年開始，人工智慧這個詞突然變得非常非常非常的熱門，講大家、政府通都在講人工智慧這個詞。

但人工智慧是什麼呢？人工智慧其實一點都不是新的詞彙，人工智慧(AI)、Artificial Intelligence這個詞彙，在1950年代就有了。那這個詞意味著什麼呢？這個詞意味著一個人類長遠以來的目標，希望機器可以跟人一樣的聰明。在科幻小說裡面，我們看要很多這樣的幻想和期待。但很長段時間裡面，人們並不知道怎麼做到人工智慧這件事情，直到後來，大概1980年代以後，有了機器學習的方法。那麼機器學習顧名思義，從名字就可以被猜出，就是讓機器具有學習的能力。所以機器學習跟人工智慧之間什麼關係呢？

人工智慧是我們想要達成的目標，而機器學習是想要達成目標的手段，希望機器通過學習方式，他跟人一樣聰明。而深度學習和機器學習有什麼關係呢？深度學習就是機器學習的其中一種方法。

在有深度學習、機器學習之前，人們用什麼樣的方式來做到人工智慧這件事呢？我記得高中生物學告訴我們說：生物的行為取決於兩件事，一個是後天學習的結果，不是後天學習的結果就是先天的本能。對於機器來說也是一樣，他怎麼樣表現的很有智慧，要么就是通過後天學習的手段表現的很有智慧，要么就是它的先天的本能。機器為什麼會有先天的本能，那可能就是他的創造者，其實都是人類，幫牠事先設立好的。

![](res/chapter1-2.png)

現在先來看一下生物的本能，講一個跟機器學習一點都沒有關係的內容：生物的本能。如圖這個是河狸，那河狸的特色呢就是它會築水壩把水擋起來。但是河狸怎麼知道要築水壩呢？河狸築水壩能力是天生的。也就是說，假設河狸他在實驗室出生，它沒有父母叫他怎麼築水壩。但是他一生下來，它心裡就有個衝動，就是它想要築水壩。那如果我們要程序語言來描述他的話，他那的程序語言就是這樣的：

![](res/chapter1-3.png)

If 它聽到流水聲

Then 它就築水壩直到他聽不到流水聲。

![](res/chapter1-4.png)

所以，生物學家就可以欺負河狸，他用一個揚聲器來播放流水聲，如果他把揚聲器流放在水泥牆裡面，然後河狸就會在水泥牆上面的放很多的樹枝，在水泥牆上面築堤，想把揚聲器的聲音蓋住。如果你把揚聲器放在地上，河狸就會用樹枝把他蓋住直到你聽不見揚聲器的聲音為止。這就是生物的本能，那機器的本能跟生物的本能其實也很像。

![](res/chapter1-5.png)

假設有一天你想要做一個chat-bot，如果你不是用機器學習的方式，而是給他天生的本能的話，那像是什麼樣子呢？你可能就會在這個chat-bot裡面，在這個聊天機器人裡面的設定一些規則，這些規則我們通常稱hand-crafted rules，叫做人設定的規則。那假設你今天要設計一個機器人，他可以幫你打開或關掉音樂，那你的做法可能是這樣：設立一條規則，就是寫程序。如果輸入的句子裡面看到“turn off”這個詞彙，那chat-bot要做的事情就是把音樂關掉。這個時候，你之後對chat-bot說，Please turn off the music 或can you turn off the music, Smart? 它就會幫你把音樂關掉。看起來好像很聰明。別人就會覺得果然這就是人工智慧。但是如果你今天想要欺負chat-bot的話，你就可以說please don‘t turn off the music，但是他還是會把音樂關掉。這是個真實的例子，你可以看看你身邊有沒有這種類似的chat-bot，然後你去真的對他說這種故意欺負它的話，它其實是會答錯的。這是真實的例子，但是不告訴他是哪家公司產品，這家公司也是號稱他們做很多AI的東西的。

![](res/chapter1-6.png)

使用hand-crafted rules有什麼樣的壞處呢，它的壞處就是：使用hand-crafted rules你沒辦法考慮到所有的可能性，它非常的僵化，而用hand-crafted rules創造出來的machine，它永遠沒有辦法超過它的創造者人類。人類想不到東西，就沒辦法寫規則，沒有寫規則，機器就不知道要怎麼辦。所以如果一個機器，它只能夠按照人類所設定好的hand-crafted rules，它整個行為都是被規定好的，沒有辦法freestyle。如果是這樣的話，它就沒有辦法超越創造他的人類。

你可能會說，但是你好像看到很多chat-bot看起來非常的聰明。如果你是有一個是一個非常大的企業，他給以派給成千上萬的工程師，用血汗的方式來建出數以萬計的規則，然後讓他的機器看起來好像很聰明。但是對於中小企業來說，這樣建規則的方式反而是不利的。所以我認為機器學習發展，對比較小規模企業反而是更有利的。因為接下來，不需要非常大量的人來幫你想各式各樣的規則，只要手上有data，你可以讓機器來幫你做這件事情。當然怎麼收集data又是另外一個問題，這不是我們今天要討論的主題

![](res/chapter1-7.png)

AI這個詞現在非常非常非常非常的熱門，所以會有各式各樣、奇奇怪怪的東西，我覺得現在非常經常碰到的一個問題，也許可用以下這個漫畫來說明，這是四格漫畫，而這個漫畫並不是隨隨便便的一個四格漫畫，這個漫畫是facebook上的漫畫。

這個漫畫想要說的是：現在你一定常常新聞或者是商場上看到這個訊息，有一個seller說看看我們最新的人工智慧機器人，它就是非常的人工智慧。這個系統搭配一個能言善道seller，加上一個非常非常潮的前端和外殼，裡面是什麼沒有人知道。

外面的觀眾就問說：他是用什麼neural方法做的，反正就是最潮的AI的技術。但是你把他剖來看一看，裡面通通都是if掉出來。

現在政府、企業都說想要推廣的AI，可是他們想要推廣AI其實是這種AI。那這個其實都不是我們現在應該做的事，如果你要推動，如果你要推廣的是這種hand-crafted AI的話，你怎麼五十年前不推廣，一直到今天才出來做呢？今天我們要走的不是這個路線，如果是這個路線是要被diss的，

![](res/chapter1-8.png)

我們要做的其實是讓機器他有自己學習的能力，也就我們要做的應該machine learning的方向。講的比較擬人化一點，所謂machine learning的方向，就是你就寫段程序，然後讓機器人變得了很聰明，他就能夠有學習的能力。接下來，你就像教一個嬰兒、教一個小孩一樣的教他，你並不是寫程序讓他做到這件事，你是寫程序讓它具有學習的能力。然後接下來，你就可以用像教小孩的方式告訴它。假設你要叫他學會做語音辨識，你就告訴它這段聲音是“Hi”，這段聲音就是“How are you”，這段聲音是“Good bye”。希望接下來它就學會了，你給它一個新的聲音，它就可以幫你產生語音辨識的結果。

![](res/chapter1-9.png)

如果你希望他學會怎麼做影像辨識，你可能不太需要改太多的程序。因為他本身就有這種學習的能力，你只是需要交換下告訴它：看到這張圖片，你要說這是猴子；看到這張圖片，然後說是貓；看到這張圖片，可以說是狗。它具有影像辨識的能力，接下來看到它之前沒有看過的貓，希望它可以認識。

![](res/chapter1-10.png)

如果講的更務實一點的話，machine learning所做的事情，你可以想成就是在尋找一個function，要讓機器具有一個能力，這種能力是根據你提供給他的資料，它去尋找出我們要尋找的function。還有很多關鍵問題都可以想成是我們就是需要一個function。



![](res/chapter1-11.png)


在語音辨識這個問題裡面，我們要找一個function，它的輸入是聲音訊號，他的輸出是語音辨識的文字。這個function非常非常的複雜，有人會想說我來用一些寫規則的方式，讀很多語言學文獻，然後寫一堆規則，然後做語音辨識。這件事情，60年代就有人做，但到現在都還沒有做出來。語音辨識太過複雜，這個function太過的複雜，不是人類所可以寫出來，這是可以想像的。所以我們需要憑藉的機器的力量，幫我們把這個function找出來。

假設你要做影像辨識，那就是找一個function，輸入一張圖片，然後輸出圖片裡面有什麼樣的東西。
或者是大家都一直在說的Alpha GO，如果你要做一個可以下圍棋machine時，其實你需要的也就是找一個function。這個function的輸入是圍棋上十九* 十九的棋盤。告訴機器在十九* 十九的棋盤上，哪些位置有黑子，哪些位置有白子。然後機器就會告​​訴你，接下來下一步應該落子在哪。或者是你要做一個聊天機器人，那你需要的是一個function，這個function的輸入就是使用者的input，它的輸出就是機器的回應。

![](res/chapter1-12.png)

以下我先很簡短的跟大家說明怎麼樣找出這個function，找出function的framework是什麼呢？我們以影像辨識為例，我們找個function輸入一張圖片，它告訴我們這個圖片裡面有什麼樣的東西。

![](res/chapter1-13.png)

在做這件事時，你的起手事是你要先準備一個function set(集合)，這個function裡面有成千上萬的function。舉例來說，這個function在裡面,有一個f1，你給它看一隻貓，它就告訴你輸出貓，看一隻狗就輸出狗。有一個function f2它很怪，你給它看貓，它說是猴子；你給他看狗，它說是蛇。你要準備一個function set，這個function set裡面有成千上萬的function。這件事情講起來可能有點抽象，你可能會懷疑說怎麼會有成千上萬的function，我怎麼把成千上萬的function收集起來，這個內容我們之後會再講。

![](res/chapter1-14.png)

總之，我們先假設你手上有一個function set，這個function set就叫做model(模型)。

![](res/chapter1-15.png)

有了這個function set，接下來機器要做的事情是：它有一些訓練的資料，這些訓練資料告訴機器說一個好的function，它的輸入輸出應該長什麼樣子，有什麼樣關係。你告訴機器說呢，現在在這個影像辨識的問題裡面，如果看到這個猴子，看到這個猴子圖也要輸出猴子，看到這個貓的圖也要輸出猴子貓，看到這個狗的圖，就要輸出猴子貓狗，這樣才是對的。只有這些訓練資料，你拿出一個function，機器就可以判斷說，這個function是好的還是不好的。

![](res/chapter1-16.png)
機器可以根據訓練資料判斷一個function是好的，還是不好的。舉例來說：在這個例子裡面顯然$f_1$，他比較符合training data的敘述，比較符合我們的知識。所以f1看起來是比較好的。 $f_2$看起來是一個荒謬的function。我們今天講的這個task叫做supervised learning。

![](res/chapter1-17.png)

如果你告訴機器input和output這就叫做supervised learning，之後我們也會講到其他不同的學習場景。現在機器有辦法決定一個function的好壞。但光能夠決定一個function的好壞是不夠的，因為在你的function set裡面，他有成千上萬的function，它有會無窮無盡的function，所以我們需要一個有效率的演算法，有效率的演算法可以從function的set裡面挑出最好的function。一個一個衡量function的好壞太花時間，實際上做不到。所以我們需要有一個好的演算法，從function set裡面挑出一個最好的的function，這個最好的function將它記為$f^*$

![](res/chapter1-18.png)

找到$f^ *$之後，我們希望用它應用到一些場景中，比如：影像辨識，輸入一張在機器沒有看過的貓，然後希望輸出也是貓。你可能會說：機器在學習時沒有看到這隻貓，那咋樣知道在測試時找到的最好function $f^ *$可以正確辨識這隻貓呢？這就是machine learning裡面非常重要的問題：機器有舉一反三的能力，這個內容後面再講。

![](res/chapter1-19.png)

左邊這個部分叫training，就是學習的過程；右邊這個部分叫做testing，學好以後你就可以拿它做應用。所以在整個machine learning framework整個過程分成了三個步驟。第一個步驟就是找一個function，第二個步驟讓machine可以衡量一個function是好還是不好，第三個步驟是讓machine有一個自動的方法，有一個好演算法可以挑出最好的function。

![](res/chapter1-20.png)

機器學習其實只有三個步驟，這三個步驟簡化了整個process。可以類比為：把大象放進冰箱。我們把大象塞進冰箱，其實也是三個步驟：把門打開；象塞進去；後把門關起來，然後就結束了。所以說，機器學習三個步驟，就好像是說把大象放進冰箱，也只需要三個步驟。


## 機器學習相關的技術

![](res/chapter1-21.png)

如圖為這學期的Learning Map，看起來是有點複雜的，我們一塊一塊來解釋，接下里我們將從圖的左上角來進行學習。

### 監督學習

![](res/chapter1-22.png)

Regression是一種machine learning的task，當我們說：我們要做regression時的意思是，machine找到的function，它的輸出是一個scalar，這個叫做regression。舉例來說，在作業一里面，我們會要你做PM2.5的預測（比如說預測明天上午的PM2.5） ，也就是說你要找一個function，這個function的輸出是未來某一個時間PM2 .5的一個數值，這個是一個regression的問題。

機器要判斷function明天上午的PM2.5輸出，你要提供給它一些資訊，它才能夠猜出明天上午的PM2.5。你給他資訊可能是今天上的PM2.5、昨天上午的PM2.5等等。這是一個function，牠吃我們給它過去PM2.5的資料，它輸出的是預測未來的PM2.5。

![](res/chapter1-23.png)

若你要訓練這種machine，如同我們在Framework中講的，你要準備一些訓練資料，什麼樣的訓練資料？你就告訴它是今天我們根據過去從政府的open data上蒐集下來的資料。九月一號上午的PM2.5是63，九月二號上午的PM2.5是65，九月三號上午的PM2.5是100。所以一個好的function輸入九月一號、九月二號的PM2.5，它應該輸出九月三號的PM2.5；若給function九月十二號的PM2.5、九月十三號的PM2.5，它應該輸出九月十四號的PM2.5。若收集更多的data，那你就可以做一個氣象預報的系統。

![](res/chapter1-24.png)

接下來講的是Classification（分類）的問題。 Regression和Classification的差別就是我們要機器輸出的東西的類型是不一樣。在Regression中機器輸出的是一個數值，在Classification裡面機器輸出的是類別。假設Classification問題分成兩種，一種叫做二分類輸出的是是或否（Yes or No）；另一類叫做多分類（Multi-class），在Multi-class中是讓機器做一個選擇題，等於是給他數個選項，每個選項都是一個類別，讓他從數個類別裡選擇正確的類別。

![](res/chapter1-25.png)

舉例來說，二分類可以鑑別垃圾郵件，將其放到垃圾箱。那怎麼做到這件事呢？其實就是需要一個function，它的輸入是一個郵件，輸出為郵件是否為垃圾郵件。

![](res/chapter1-26.png)

你要訓練這樣的function很簡單，給他一大堆的Data並告訴它，現在輸入這封郵件，你應該說是垃圾郵件，輸入這封郵件，應該說它不是垃圾郵件。你給他夠多的這種資料去學，它就可以自動找出一個可以偵測垃圾郵件的function。

![](res/chapter1-27.png)

多分類的舉一個文章分類的例子，現在網絡上有非常非非常多的新聞，也許沒有人會把所有的新聞看完，但希望機器自動幫一把新聞做分類。怎麼做呢？你需要的是一個function，它的輸入是一則新聞，輸出是新聞屬於哪個類別，你要做的事情就是解這個選擇題。

若要訓練這種機器就要準備很多訓練資料（Training Data），然後給它新的文章，新聞它能給你正確的結果。

![](res/chapter1-28.png)

剛才講的都是讓machine去解的任務，接下來要講的是在解任務的過程中第一步就是要選擇function set，選不同的function set就是選不同的model。 Model有很多種，最簡單的就是線性模型，但我們會花很多時間在非線性的模型上。在非線性的模型中最耳熟能詳的就是Deep learning。

![](res/chapter1-29.png)

在做Deep learning時，它的function是特別複雜的，所以它可以做特別複雜的事情。比如它可以做影像辨識，這個複雜的function可以描述pixel和class之間的關係。

![](res/chapter1-30.png)

用Deep learning的技術也可以讓機器下圍棋，
下圍棋這個task 其實就是一個分類的問題。對分類問題我們需要一個很複雜的function，輸入是一個棋盤的格子，輸出就是下一步應該落子的位置。我們知道一個棋盤上有十九乘十九的位置可以落子，所以今天下圍棋這件事情，你就可以把它想成是一個十九乘十九個類別的分類問題，或者是你可以把它想成是一個有十九乘十九個選項的選擇題。

![](res/chapter1-31.png)

你要怎麼訓練機器讓他學會下圍棋呢？你要蒐集訓練資料，告訴機器現在這個function輸入輸出分別應該是什麼。就看到某樣的盤式，我們應該輸出什麼樣結果。

怎麼收集資料呢？你可以從人類過去下的棋庫裡面蒐集。舉例來說，你收集了進藤光和社新春下的那一盤棋的棋譜。社新春出手先下五之5，進藤光次手下天元，社新春第三手下五之5。

![](res/chapter1-32.png)

所以若你有了這樣的棋譜之後，可以告訴machine如果現在有人落子下5之五，下一步就落子在天元；若五之五和天元都有落子，那就要落子在另外一個五之5上。然後你給它足夠多的棋譜，他就學會下圍棋了。

![](res/chapter1-33.png))

除了deep learning 以外還有很多machine learning的model也是非線性的模型，這學期會請吳佩雲老師來幫我們講SVM。

### 半監督學習

![](res/chapter1-34.png)

剛才我們講的都是supervised learning（監督學習），監督學習的問題是我們需要大量的training data。 training data告訴我們要找的function的input和output之間的關係。如果我們在監督學習下進行學習，我們需要告訴機器function的input和output是什麼。這個output往往沒有辦法用很自然的方式取得，需要人工的力量把它標註出來，這些function的output叫做label。

那有沒有辦法減少label需要的量呢？就是半監督學習。

![](res/chapter1-35.png)

假設你先想讓機器鑑別貓狗的不同。你想做一個分類器讓它告訴你，圖片上是貓還是狗。你有少量的貓和狗的labelled data，但是同時你又有大量的Unlabeled data，但是你沒有力氣去告訴機器說哪些是貓哪些是狗。在半監督學習的技術中，這些沒有label的data，他可能也是對學習有幫助。這個我們之後會講為什麼這些沒有label的data對學習會有幫助。


### 遷移學習

![](res/chapter1-36.png)

另外一個減少data用量的方向是遷移學習。

![](res/chapter1-37.png)

遷移學習的意思是：假設我們要做貓和狗的分類問題，我們也一樣，只有少量的有label的data。但是我們現在有大量的data，這些大量的data中可能有label也可能沒有label。但是他跟我們現在要考慮的問題是沒有什麼特別的關係的，我們要分辨的是貓和狗的不同，但是這邊有一大堆其他動物的圖片還是動畫圖片（涼宮春日，禦坂美琴）你有這一大堆不相干的圖片，它到底可以帶來什麼幫助。這個就是遷移學習要講的問題。


### 無監督學習

![](res/chapter1-38.png)

更加進階的就是無監督學習，我們希望機器可以學到無師自通。

![](res/chapter1-39.png)

如果在完全沒有任何label的情況下，到底機器可以學到什麼樣的事情。舉例來說，如果我們給機器看大量的文章（在去網絡上收集站文章很容易，網絡上隨便爬就可以）讓機器看過大量的文章以後，它到底可以學到什麼事情。

![](res/chapter1-40.png)

它能不能夠學會每一個詞彙的意思，要讓機器學會每一個詞彙的意思，你可以想成是我們找一個function，然後你把一個詞彙丟進去。比如說你把“apple”丟進這個function裡面，機器要輸出告訴你說這個詞會是什麼意思。也許他用一個向量來表示這個詞彙的各種不同的特性。但現在講是無監督學習的問題，你現在只有一大堆的文章，也就是說你只有詞彙，你只有function的輸入，沒有任何的輸出。那你到底要怎麼解決這個問題。

![](res/chapter1-41.png)

我們舉另外一個無監督學習的例子：假設我們今天帶機器去動物園讓它看一大堆的動物，它能不能夠在看了一大堆動物以後，它就學會自己創造一些動物。那這個都是真實例子。仔細看了大量的動物以後，它就可以自己的畫一些狗出來。有眼睛長在身上的狗、還有乳牛狗等等。

![](res/chapter1-42.png)

這個Task也是一個無監督學習的問題，這個function的輸入不知道是什麼，可能是某一個code代表要輸出圖片的特性，輸出是一張圖片。你給機器看到的只有非常大量的圖片，只有function的input，沒有output。機器要咋樣生成新的圖片，這是我們後面要解決的問題。


### 監督學習中的結構化學習

![](res/chapter1-43.png)

在machine要解的任務上我們講了Regression、classification，還有一類的問題是structured learning。

![](res/chapter1-44.png)

structured learning 中讓機器輸出的是要有結構性的，舉例來說：在語音辨識裡面，機器輸入是聲音訊號，輸出是一個句子。句子是要很多詞彙拼湊完成。它是一個有結構性的object。或者是說在機器翻譯裡面你說一句話，你輸入中文希望機器翻成英文，它的輸出也是有結構性的。或者你今天要做的是人臉辨識，來給機器看張圖片，它會知道說最左邊是長門，中間是涼宮春日，右邊是寶玖瑠。然後機器要把這些東西標出來，這也是一個structure learning問題。

![](res/chapter1-45.png)

 其實多數人可能都聽過regression，也聽過classification，你可能不見得聽過structure learning。很多教科書都直接說，machine learning是兩大類的問題，regression，和classification。 machine learning只有regression和classification兩類問題，就好像告訴你：我們所熟知的世界只有五大洲，但是這只是真實世界的一小部分，真正的世界是如圖所示。
 
真正世界還應該包括structure learning，這裡面還有很多問題是沒有探究的。

![](res/chapter1-46.png)

最後一部分就是reinforcement learning的問題。

![](res/chapter1-47.png)

reinforcement learning其實是一個已經發展了很久的技術，但近期受到大家的關注是因為data mining將reinforcement learning技術用來玩一些小遊戲。另外一個就是Alpha Go。

### 強化學習

![](res/chapter1-48.png)

我們若將強化學習和監督學習進行比較時，在監督學習中我們會告訴機器正確答案是什麼。若現在我們要用監督學習的方法來訓練一個聊天機器人，你的訓練方式會是：你就告訴機器，現在使用者說了hello，你就說hi，現在使用者說了byebye ，你就說good bye。所以機器有一個人當他家教在他旁邊手把手的教他每件事情，這就是監督學習。

reinforcement learning是什麼呢？在reinforcement learning裡面，我們沒有告訴機器正確的答案是什麼，機器所擁有的只有一個分數，就是他做的好還是不好。若我們現在要用reinforcement learning方法來訓練一個聊天機器人的話，他訓練的方法會是這樣：你就把機器發到線下，讓他的和麵進來的客人對話，然後想了半天以後呢，最後仍舊勃然大怒把電話掛掉了。那機器就學到一件事情就是剛才做錯了。但是他不知道哪邊錯了，它就要回去自己想道理，是一開始就不應該打招呼嗎？還是中間不應該在罵髒話了之類。它不知道，也沒有人告訴它哪裡做的不好，它要回去反省檢討哪一步做的不好。機器要在reinforcement learning的情況下學習，機器是非常intelligence的。 reinforcement learning也是比較符合我們人類真正的學習的情景，這是你在學校裡面的學習老師會告訴你答案，但在真實社會中沒人回告訴你正確答案。你只知道你做得好還是做得不好，如果機器可以做到reinforcement learning，那確實是比較intelligence。

![](res/chapter1-49.png)

若我們用Alpha Go當做例子時，supervised learning就是告訴機器：看到這個盤式你就下“5-5”，看到這個盤式你就下“3-3”

reinforcement learning的意思是：機器跟對手互下，機器會不斷的下棋，最後贏了，機器就會知道下的不錯，但是究竟是哪裡可以使它贏，它其實是不知道的。我們知道Alpha Go其實是用監督學習加上reinforcement learning去學習的。先用棋譜做監督學習，然後在做reinforcement learning，但是reinforcement learning需要一個對手，如果使用人當對手就會很讓費時間，所以機器的對手是另外一個機器。



### 小提醒

![](res/chapter1-50.png)

 大家注意一下這個不同的方塊，我是用不同的顏色來表示。同樣的顏色不同的方塊是同一個類型的，這邊的藍色的方塊，指的是學習的情景，通常學習的情景是你沒有辦法控制的。比如，因為我們沒有data做監督學習，所以我們才做reinforcement learning。現在因為Alpha Go比較火，所以Alpha Go中用到的reinforcement learning會被認為比較潮。所以說有學生去面試，說明自己是做監督學習的，就會被質疑為什麼不做reinforcement learning。那這個時候你就應該和他說，如果我今天可以監督學習，其實就不應該做reinforcement learning。 reinforcement learning就是我們沒有辦法做監督學習的時候，我們才做reinforcement learning。紅色的是指你的task，你要解的問題，你要解的這個問題隨著你用的方程的不同，有regression、有classification、有structured。所以在不同的情境下，都有可能要解這個task。最後，在這些不同task裡面有不同的model，用綠色的方塊表示。


