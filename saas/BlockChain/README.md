
## V1.Docker+BlockChain+Microservices

https://www.ibm.com/developerworks/library/j-chaincode-for-java-developers/index.html

http://radiostud.io/eight-blockchain-platforms-comparison/

Chaincode, also called the smart contract, is essentially the business logic that governs how the different entities or parties in a blockchain network interact or transact with each other. Simply put, the chaincode is the encapsulation of business network transactions in code. Invocations of the chaincode result in sets and gets of the ledger or world state.

https://www.ibm.com/developerworks/cloud/library/cl-ibm-blockchain-chaincode-development-using-golang/index.html

### Data models in chaincode

The Hyperledger ledger consists of two parts:

#### World state, 

which is stored in a key value store. This key value store is powered by the RocksDB. This key value store takes in a byte array as the value, which can be used to store a serialized JSON structure. Essentially this key value store can be used to store any custom data model/schema required by your smart contract to function.

#### Blockchain, 

which consists of a series of blocks each containing a number of transactions. Each block contains the hash of the world state and is also linked to the previous block. Blockchain is append-only.

### What happened exactly on chaincode(smart contract) deploy and invoke , query, in Hyperledger?


#### Deploy

During “Deploy” the chain code is submitted to the ledger in a form of transaction and distributed to all nodes in the network. Each node creates new Docker container with this chaincode embedded. After that container will be started and Init method will be executed.

1.Make sure you can compile your code locally, 

2.place your code in a public github repo, 

3.submit a deploy request,

4.check the response for any errors.

#### Query

During “Query” - chain code will read the current state and send it back to user. This transaction is not saved in blockchain.

#### Invoke

During “Invoke” - chaincode can modify the state of the variables in ledger. Each “Invoke” transaction will be added to the “block” in the ledger.


### 基本原理

区块链的基本原理,基本概念包括：

交易（Transaction）：一次操作，导致账本状态的一次改变，如添加一条记录；

区块（Block）：记录一段时间内发生的交易和状态结果，是对当前账本状态的一次共识；

链（Chain）：由一个个区块按照发生顺序串联而成，是整个状态变化的日志记录。

如果把区块链作为一个状态机，则每次交易就是试图改变一次状态，而每次共识生成的区块，就是参与者对于区块中所有交易内容导致状态改变的结果进行确认。

https://github.com/yeasy/blockchain_guide/blob/master/born/what.md
